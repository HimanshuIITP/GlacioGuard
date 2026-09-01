#!/usr/bin/env python3
"""
GlacioGuard — Temporal Alignment
Aligns all observations using asof merge to prevent temporal leakage.
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import yaml
from datetime import datetime, timezone

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PROCESSED = BASE_DIR / "data" / "processed"
OBS_CONFIG = BASE_DIR / "config" / "observation_config.yaml"

def get_config():
    if not OBS_CONFIG.exists():
        return {}
    with open(OBS_CONFIG, "r") as f:
        return yaml.safe_load(f)

def build_reference_grid(lakes, start, end, freq="1D"):
    """Create the master daily grid for each lake."""
    try:
        dt_start = pd.to_datetime(start).tz_localize('UTC')
        dt_end = pd.to_datetime(end).tz_localize('UTC')
    except Exception:
        # Fallback if already tz-aware
        dt_start = pd.to_datetime(start)
        dt_end = pd.to_datetime(end)
        
    dates = pd.date_range(dt_start, dt_end, freq=freq)
    
    idx = pd.MultiIndex.from_product([lakes, dates], names=["lake_uid", "reference_timestamp"])
    df = pd.DataFrame(index=idx).reset_index()
    
    # Sort carefully for asof merge
    df = df.sort_values(by=["lake_uid", "reference_timestamp"]).reset_index(drop=True)
    df["reference_timestamp"] = df["reference_timestamp"].astype('datetime64[ns, UTC]')
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--lake-id", type=str)
    
    args, _ = ap.parse_known_args()
    
    print("\n" + "=" * 60)
    print("  STAGE: Temporal Alignment")
    print("=" * 60)
    
    config = get_config()
    start = config.get("development", {}).get("start_date", "2020-01-01")
    end = config.get("development", {}).get("end_date") or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    freq = config.get("temporal", {}).get("reference_frequency", "1D")
    
    # Get test lakes or all
    gj_path = DATA_PROCESSED / "india_glacial_lakes_2022.geojson"
    if not gj_path.exists():
        print("No lake geometry found.")
        return
        
    import geopandas as gpd
    gdf = gpd.read_file(gj_path)
    
    if args.lake_id:
        uids = [args.lake_id]
    elif args.test:
        with open(BASE_DIR / "config" / "test_lakes.yaml", "r") as f:
            uids = yaml.safe_load(f).get("test_lakes", [])
    else:
        uids = gdf["lake_uid"].unique()
        
    print(f"  Building {freq} master grid from {start} to {end}...")
    master = build_reference_grid(uids, start, end, freq)
    
    # Provenance columns to exclude from all source merges (avoid collisions)
    _PROVENANCE_COLS = {"source", "source_product", "source_version", "quality_flag",
                        "quality_reason", "processing_version", "retrieved_at"}

    # Merge Terrain (static)
    terrain_path = DATA_PROCESSED / "lake_terrain.parquet"
    if terrain_path.exists():
        df_t = pd.read_parquet(terrain_path)
        if not df_t.empty:
            # Keep DEM-specific prefixed columns, drop generic provenance
            df_t = df_t.drop(columns=[c for c in _PROVENANCE_COLS if c in df_t.columns])
            master = master.merge(df_t, on="lake_uid", how="left")
            
    # Merge Weather (time series) using asof
    wx_path = DATA_PROCESSED / "lake_weather_features.parquet"
    wx_status = "SKIPPED"
    if wx_path.exists():
        df_w = pd.read_parquet(wx_path)
        if not df_w.empty:
            wx_status = "SUCCESS"
            if "observation_timestamp" not in df_w.columns and "timestamp" in df_w.columns:
                df_w = df_w.rename(columns={"timestamp": "observation_timestamp"})
            df_w["observation_timestamp"] = pd.to_datetime(df_w["observation_timestamp"])
            if df_w["observation_timestamp"].dt.tz is None:
                df_w["observation_timestamp"] = df_w["observation_timestamp"].dt.tz_localize('UTC')
            df_w["observation_timestamp"] = df_w["observation_timestamp"].astype('datetime64[ns, UTC]')
                
            df_w = df_w.sort_values(by="observation_timestamp")
            
            # asof merge per lake
            merged_pieces = []
            for uid, group in master.groupby("lake_uid"):
                w_group = df_w[df_w["lake_uid"] == uid]
                if not w_group.empty:
                    res = pd.merge_asof(
                        group, w_group.drop(columns=["lake_uid"]),
                        left_on="reference_timestamp",
                        right_on="observation_timestamp",
                        direction="backward"  # CRITICAL for no leakage
                    )
                    merged_pieces.append(res)
                else:
                    merged_pieces.append(group)
                    
            master = pd.concat(merged_pieces, ignore_index=True)
            
    # Compute Observation Age
    if "observation_timestamp" in master.columns:
        master["weather_observation_age_hours"] = (master["reference_timestamp"] - master["observation_timestamp"]).dt.total_seconds() / 3600.0
        master["actual_weather_observation"] = master["weather_observation_age_hours"].fillna(float('inf')) <= 24
        
        max_wx_age = config.get("temporal", {}).get("max_weather_age_hours", 48)
        master["weather_stale"] = master["weather_observation_age_hours"] > max_wx_age
        
        wx_cols = [c for c in df_w.columns if c not in ["lake_uid", "observation_timestamp"]]
        master.loc[master["weather_stale"], wx_cols] = np.nan
        
        # Rename so we don't conflict with other merges
        master = master.rename(columns={
            "observation_timestamp": "weather_timestamp",
            "run_id": "weather_run_id",
            "observation_id": "weather_observation_id"
        })
        
    # Merge Satellite
    sat_path = DATA_PROCESSED / "lake_satellite_features.parquet"
    sat_status = "SKIPPED"
    if sat_path.exists():
        df_s = pd.read_parquet(sat_path)
        if not df_s.empty:
            sat_status = "SUCCESS"
            if "observation_timestamp" not in df_s.columns and "timestamp" in df_s.columns:
                df_s = df_s.rename(columns={"timestamp": "observation_timestamp"})
            df_s["observation_timestamp"] = pd.to_datetime(df_s["observation_timestamp"], utc=True).astype('datetime64[ns, UTC]')
            df_s = df_s.sort_values(by="observation_timestamp")
            
            merged_pieces = []
            # Drop provenance columns to avoid merge conflicts
            sat_drop = ["lake_uid"] + [c for c in _PROVENANCE_COLS if c in df_s.columns]
            for uid, group in master.groupby("lake_uid"):
                s_group = df_s[df_s["lake_uid"] == uid]
                if not s_group.empty:
                    res = pd.merge_asof(
                        group, s_group.drop(columns=sat_drop),
                        left_on="reference_timestamp",
                        right_on="observation_timestamp",
                        direction="backward"
                    )
                    merged_pieces.append(res)
                else:
                    merged_pieces.append(group)
            master = pd.concat(merged_pieces, ignore_index=True)
            
            if "observation_timestamp" in master.columns:
                master["satellite_observation_age_hours"] = (master["reference_timestamp"] - master["observation_timestamp"]).dt.total_seconds() / 3600.0
                master["actual_satellite_observation"] = master["satellite_observation_age_hours"].fillna(float('inf')) <= 24
                
                max_sat_age = config.get("temporal", {}).get("max_satellite_age_hours", 720)
                master["satellite_stale"] = master["satellite_observation_age_hours"] > max_sat_age
                sat_cols = [c for c in df_s.columns if c not in ["lake_uid", "observation_timestamp"]]
                master.loc[master["satellite_stale"], sat_cols] = np.nan
                
                master = master.rename(columns={
                    "observation_timestamp": "satellite_timestamp",
                    "run_id": "satellite_run_id",
                    "observation_id": "satellite_observation_id"
                })
                
    # Merge Snow
    snow_path = DATA_PROCESSED / "lake_snow_observations.parquet"
    snow_status = "SKIPPED"
    if snow_path.exists():
        df_sn = pd.read_parquet(snow_path)
        if not df_sn.empty:
            snow_status = "SUCCESS"
            if "observation_timestamp" not in df_sn.columns and "timestamp" in df_sn.columns:
                df_sn = df_sn.rename(columns={"timestamp": "observation_timestamp"})
            df_sn["observation_timestamp"] = pd.to_datetime(df_sn["observation_timestamp"], utc=True).astype('datetime64[ns, UTC]')
            df_sn = df_sn.sort_values(by="observation_timestamp")
            
            merged_pieces = []
            # Drop provenance columns to avoid merge conflicts
            snow_merge_drop = ["lake_uid"] + [c for c in df_sn.columns if c in [
                "source", "source_product", "source_version", "quality_flag",
                "quality_reason", "processing_version", "retrieved_at"
            ]]
            for uid, group in master.groupby("lake_uid"):
                sn_group = df_sn[df_sn["lake_uid"] == uid]
                if not sn_group.empty:
                    res = pd.merge_asof(
                        group, sn_group.drop(columns=[c for c in snow_merge_drop if c in sn_group.columns]),
                        left_on="reference_timestamp",
                        right_on="observation_timestamp",
                        direction="backward"
                    )
                    merged_pieces.append(res)
                else:
                    merged_pieces.append(group)
            master = pd.concat(merged_pieces, ignore_index=True)
            
            if "observation_timestamp" in master.columns:
                master["snow_observation_age_hours"] = (master["reference_timestamp"] - master["observation_timestamp"]).dt.total_seconds() / 3600.0
                master["actual_snow_observation"] = master["snow_observation_age_hours"].fillna(float('inf')) <= 24
                
                max_snow_age = config.get("temporal", {}).get("max_snow_age_hours", 240)
                master["snow_stale"] = master["snow_observation_age_hours"] > max_snow_age
                snow_cols = [c for c in df_sn.columns if c not in ["lake_uid", "observation_timestamp"]]
                master.loc[master["snow_stale"], snow_cols] = np.nan
                
                master = master.rename(columns={
                    "observation_timestamp": "snow_timestamp",
                    "run_id": "snow_run_id",
                    "observation_id": "snow_observation_id"
                })

    # Merge GPM
    gpm_path = DATA_PROCESSED / "lake_precipitation_highfreq.parquet"
    gpm_status = "SKIPPED"
    if gpm_path.exists():
        df_g = pd.read_parquet(gpm_path)
        if not df_g.empty:
            gpm_status = "SUCCESS"
            if "observation_timestamp" not in df_g.columns and "timestamp" in df_g.columns:
                df_g = df_g.rename(columns={"timestamp": "observation_timestamp"})
            df_g["observation_timestamp"] = pd.to_datetime(df_g["observation_timestamp"], utc=True).astype('datetime64[ns, UTC]')
            df_g = df_g.sort_values(by="observation_timestamp")
            df_g = df_g.rename(columns={"precipitation_mm": "gpm_precipitation_mm"})
            
            merged_pieces = []
            for uid, group in master.groupby("lake_uid"):
                g_group = df_g[df_g["lake_uid"] == uid]
                if not g_group.empty:
                    res = pd.merge_asof(
                        group, g_group.drop(columns=["lake_uid", "source", "source_product", "source_version", "quality_flag", "quality_reason", "processing_version", "retrieved_at"]),
                        left_on="reference_timestamp",
                        right_on="observation_timestamp",
                        direction="backward"
                    )
                    merged_pieces.append(res)
                else:
                    merged_pieces.append(group)
            master = pd.concat(merged_pieces, ignore_index=True)
            
            if "observation_timestamp" in master.columns:
                master["gpm_observation_age_hours"] = (master["reference_timestamp"] - master["observation_timestamp"]).dt.total_seconds() / 3600.0
                master["actual_gpm_observation"] = master["gpm_observation_age_hours"].fillna(float('inf')) <= 24
                
                max_gpm_age = config.get("temporal", {}).get("max_gpm_age_hours", 48)
                master["gpm_stale"] = master["gpm_observation_age_hours"] > max_gpm_age
                gpm_cols = [c for c in df_g.columns if c not in ["lake_uid", "observation_timestamp", "source", "source_product", "source_version", "quality_flag", "quality_reason", "processing_version", "retrieved_at"]]
                master.loc[master["gpm_stale"], gpm_cols] = np.nan
                
                master = master.rename(columns={
                    "observation_timestamp": "gpm_timestamp",
                    "run_id": "gpm_run_id",
                    "observation_id": "gpm_observation_id"
                })
                
    out_path = DATA_PROCESSED / "lake_observations.parquet"
    master.to_parquet(out_path, index=False)
    print(f"  Aligned dataset saved with {len(master)} rows.")
    
    if wx_status == "SUCCESS" or sat_status == "SUCCESS" or snow_status == "SUCCESS" or gpm_status == "SUCCESS":
        print("  [STATUS] SUCCESS")
    else:
        print("  [WARN] No time-varying sources available for temporal alignment.")
        print("  [STATUS] FAILED")

if __name__ == "__main__":
    main()
