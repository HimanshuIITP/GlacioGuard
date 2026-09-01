#!/usr/bin/env python3
"""
GlacioGuard — Baselines
Calculates historical baselines safely using explicit minimum sample counts.
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PROCESSED = BASE_DIR / "data" / "processed"

MIN_SAMPLES_WX = 30
MIN_SAMPLES_SAT = 5

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--lake-id", type=str)
    
    args, _ = ap.parse_known_args()
    
    print("\n" + "=" * 60)
    print("  STAGE: Baseline Calculation")
    print("=" * 60)
    
    gj_path = DATA_PROCESSED / "india_glacial_lakes_2022.geojson"
    if gj_path.exists():
        import geopandas as gpd
        gdf = gpd.read_file(gj_path)
        if args.test:
            # test_lakes.yaml reading would go here, but for simplicity we rely on the lake_uids in the features
            pass
        elif args.lake_id:
            gdf = gdf[gdf["lake_uid"] == args.lake_id]
            
        uids = gdf["lake_uid"].unique()
    else:
        print("  [ERROR] Missing geojson.")
        print("  [STATUS] FAILED")
        return
        
    records = []
    
    # Load feature tables if they exist
    df_wx = pd.DataFrame()
    wx_feat = DATA_PROCESSED / "lake_weather_features.parquet"
    if wx_feat.exists():
        df_wx = pd.read_parquet(wx_feat)
        
    df_sat = pd.DataFrame()
    sat_feat = DATA_PROCESSED / "lake_satellite_features.parquet"
    if sat_feat.exists():
        df_sat = pd.read_parquet(sat_feat)
        
    calculated_baselines = 0
    unavailable_baselines = 0
    insufficient_sample_baselines = 0
    
    for uid in uids:
        rec = {
            "lake_uid": uid,
            "baseline_area_km2": pd.NA,
            "area_std_km2": pd.NA,
            "baseline_temperature_c": pd.NA,
            "temperature_std_c": pd.NA,
            "baseline_precipitation_mm": pd.NA,
            "baseline_period_start": pd.NA,
            "baseline_period_end": pd.NA,
            "sample_count_wx": 0,
            "sample_count_sat": 0,
            "baseline_quality_flag": "insufficient_data"
        }
        
        has_wx = False
        has_sat = False
        
        if not df_wx.empty and uid in df_wx["lake_uid"].values:
            uid_wx = df_wx[df_wx["lake_uid"] == uid]
            uid_wx = uid_wx.dropna(subset=["temperature_2m_c"])
            if len(uid_wx) >= MIN_SAMPLES_WX:
                rec["baseline_temperature_c"] = float(uid_wx["temperature_2m_c"].mean())
                rec["temperature_std_c"] = float(uid_wx["temperature_2m_c"].std())
                rec["baseline_precipitation_mm"] = float(uid_wx["precipitation_mm"].mean())
                
                # If observation_timestamp exists, use it, else fallback to timestamp
                ts_col = "observation_timestamp" if "observation_timestamp" in uid_wx.columns else "timestamp"
                rec["baseline_period_start"] = uid_wx[ts_col].min().isoformat()
                rec["baseline_period_end"] = uid_wx[ts_col].max().isoformat()
                rec["sample_count_wx"] = len(uid_wx)
                has_wx = True
                
        if not df_sat.empty and uid in df_sat["lake_uid"].values:
            uid_sat = df_sat[df_sat["lake_uid"] == uid]
            uid_sat = uid_sat.dropna(subset=["lake_area_km2"])
            if len(uid_sat) >= MIN_SAMPLES_SAT:
                rec["baseline_area_km2"] = float(uid_sat["lake_area_km2"].mean())
                rec["area_std_km2"] = float(uid_sat["lake_area_km2"].std())
                rec["sample_count_sat"] = len(uid_sat)
                
                ts_col = "observation_timestamp" if "observation_timestamp" in uid_sat.columns else "timestamp"
                sat_start = uid_sat[ts_col].min().isoformat()
                sat_end = uid_sat[ts_col].max().isoformat()
                
                if pd.isna(rec["baseline_period_start"]) or sat_start < rec["baseline_period_start"]:
                    rec["baseline_period_start"] = sat_start
                if pd.isna(rec["baseline_period_end"]) or sat_end > rec["baseline_period_end"]:
                    rec["baseline_period_end"] = sat_end
                    
                has_sat = True
                
        if has_wx or has_sat:
            rec["baseline_quality_flag"] = "good"
            calculated_baselines += 1
        else:
            if rec["sample_count_wx"] > 0 or rec["sample_count_sat"] > 0:
                insufficient_sample_baselines += 1
            else:
                unavailable_baselines += 1
                
        records.append(rec)
        
    df_base = pd.DataFrame(records)
    out_path = DATA_PROCESSED / "lake_baselines.parquet"
    
    if calculated_baselines > 0:
        if out_path.exists() and not args.all:
            existing_df = pd.read_parquet(out_path)
            existing_df = existing_df[~existing_df.lake_uid.isin(df_base.lake_uid)]
            df_base = pd.concat([existing_df, df_base], ignore_index=True)
            
        df_base.to_parquet(out_path, index=False)
        print(f"  [METRICS] calculated_baselines={calculated_baselines}")
        print(f"  [METRICS] insufficient_sample_baselines={insufficient_sample_baselines}")
        print(f"  [METRICS] unavailable_baselines={unavailable_baselines}")
        
        if unavailable_baselines > 0 or insufficient_sample_baselines > 0:
            print("  [STATUS] PARTIAL")
        else:
            print("  [STATUS] SUCCESS")
    else:
        print(f"  [METRICS] calculated_baselines={calculated_baselines}")
        print(f"  [METRICS] insufficient_sample_baselines={insufficient_sample_baselines}")
        print(f"  [METRICS] unavailable_baselines={unavailable_baselines}")
        print("  [WARN] No baselines could be calculated due to missing data.")
        print("  [STATUS] SKIPPED")

if __name__ == "__main__":
    main()
