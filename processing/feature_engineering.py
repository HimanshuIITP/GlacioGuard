#!/usr/bin/env python3
"""
GlacioGuard — Feature Engineering
Calculates time-aware rolling features for weather, satellite, and snow data.
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PROCESSED = BASE_DIR / "data" / "processed"

def engineer_weather(df):
    """Calculate rolling weather features."""
    if df.empty:
        return df
        
    df = df.copy()
    # Ensure sorted by time
    if "observation_timestamp" in df.columns:
        ts_col = "observation_timestamp"
    else:
        ts_col = "timestamp"
        
    df[ts_col] = pd.to_datetime(df[ts_col], utc=True)
    df = df.sort_values(by=["lake_uid", ts_col]).reset_index(drop=True)
    df = df.set_index(ts_col)
    
    results = []
    for uid, group in df.groupby("lake_uid"):
        # Temperature (mean/min/max over 3, 7, 30 days)
        # Using centered=False ensures we only look backward (no leakage)
        group["temperature_mean_3d"] = group["temperature_2m_c"].rolling("3D").mean()
        group["temperature_mean_7d"] = group["temperature_2m_c"].rolling("7D").mean()
        group["temperature_mean_30d"] = group["temperature_2m_c"].rolling("30D").mean()
        
        group["temperature_min_3d"] = group["temperature_2m_c"].rolling("3D").min()
        group["temperature_max_3d"] = group["temperature_2m_c"].rolling("3D").max()
        
        # Time-aware difference using merge_asof
        group_reset = group.reset_index()
        for window in ["1D", "7D"]:
            offset = pd.to_timedelta(window)
            group_reset[f"target_ts_{window}"] = group_reset[ts_col] - offset
            
            temp = pd.merge_asof(
                group_reset[[ts_col, f"target_ts_{window}", "temperature_2m_c"]].sort_values(f"target_ts_{window}"),
                group_reset[[ts_col, "temperature_2m_c"]].sort_values(ts_col),
                left_on=f"target_ts_{window}",
                right_on=ts_col,
                direction="backward",
                suffixes=("", "_past")
            )
            temp = temp.sort_values(ts_col).reset_index(drop=True)
            group_reset[f"temperature_change_{window.lower()}"] = group_reset["temperature_2m_c"] - temp["temperature_2m_c_past"]
            group_reset = group_reset.drop(columns=[f"target_ts_{window}"])
            
        group = group_reset.set_index(ts_col)
        
        # Precipitation (sums)
        group["precipitation_24h"] = group["precipitation_mm"].rolling("1D").sum()
        group["precipitation_72h"] = group["precipitation_mm"].rolling("3D").sum()
        group["precipitation_7d"] = group["precipitation_mm"].rolling("7D").sum()
        group["precipitation_30d"] = group["precipitation_mm"].rolling("30D").sum()
        
        # Snowfall
        group["snowfall_7d"] = group["snowfall_mm"].rolling("7D").sum()
        group["snowfall_30d"] = group["snowfall_mm"].rolling("30D").sum()
        
        results.append(group.reset_index())
        
    return pd.concat(results, ignore_index=True)

def engineer_satellite(df):
    """Calculate lake area changes using actual time offsets."""
    if df.empty or "lake_area_km2" not in df.columns:
        return df
        
    df = df.copy()
    # Ensure observation_timestamp is a datetime
    df["observation_timestamp"] = pd.to_datetime(df["observation_timestamp"], utc=True)
    df = df.sort_values(by=["lake_uid", "observation_timestamp"]).reset_index(drop=True)
    
    results = []
    
    from scipy.stats import linregress
    
    for uid, group in df.groupby("lake_uid"):
        group = group.copy()
        
        # Calculate changes over strict time offsets
        for window in ["7D", "30D", "90D", "365D"]:
            # Create a target timestamp column shifted back by the window
            offset = pd.to_timedelta(window)
            group[f"target_ts_{window}"] = group["observation_timestamp"] - offset
            
            # Use merge_asof to find the latest valid observation at or before target_ts
            # Sort is required for asof
            temp = pd.merge_asof(
                group[["observation_timestamp", f"target_ts_{window}", "lake_area_km2"]].sort_values(f"target_ts_{window}"),
                group[["observation_timestamp", "lake_area_km2"]].sort_values("observation_timestamp"),
                left_on=f"target_ts_{window}",
                right_on="observation_timestamp",
                direction="backward",
                suffixes=("", "_past")
            )
            # Restore original sorting
            temp = temp.sort_values("observation_timestamp").reset_index(drop=True)
            
            # Calculate change
            group[f"lake_area_change_{window.lower()}"] = group["lake_area_km2"] - temp["lake_area_km2_past"]
            group = group.drop(columns=[f"target_ts_{window}"])
            
        # Calculate trends (regression slope) over 30d and 90d
        group["lake_area_trend_30d"] = np.nan
        group["lake_area_trend_90d"] = np.nan
        
        group_indexed = group.set_index("observation_timestamp")
        
        # Iterate to compute rolling regression
        # (Using a simple loop since observations are sparse)
        for ts, row in group_indexed.iterrows():
            for w_days, w_name in [(30, "30d"), (90, "90d")]:
                start_ts = ts - pd.Timedelta(days=w_days)
                # Select window [start_ts, ts]
                window_data = group_indexed.loc[start_ts:ts]
                
                # Drop NA areas
                window_data = window_data.dropna(subset=["lake_area_km2"])
                
                if len(window_data) >= 3:
                    # Convert index to days relative to start for regression
                    x = (window_data.index - start_ts).total_seconds() / 86400.0
                    y = window_data["lake_area_km2"].values
                    try:
                        slope, intercept, r_value, p_value, std_err = linregress(x, y)
                        # Slope is change in area per day
                        group.loc[group["observation_timestamp"] == ts, f"lake_area_trend_{w_name}"] = slope
                    except Exception:
                        pass
                        
        results.append(group)
        
    if results:
        return pd.concat(results, ignore_index=True)
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--lake-id", type=str)
    
    args, _ = ap.parse_known_args()
    
    print("\n" + "=" * 60)
    print("  STAGE: Feature Engineering")
    print("=" * 60)
    
    wx_path = DATA_PROCESSED / "lake_weather_observations.parquet"
    wx_status = "SKIPPED"
    if wx_path.exists():
        print("  Processing weather features...")
        df_wx = pd.read_parquet(wx_path)
        if not df_wx.empty:
            df_wx_feat = engineer_weather(df_wx)
            df_wx_feat.to_parquet(DATA_PROCESSED / "lake_weather_features.parquet", index=False)
            wx_status = "SUCCESS"
        else:
            print("  [WARN] lake_weather_observations.parquet is empty.")
    else:
        print("  [WARN] lake_weather_observations.parquet missing.")
        
    sat_path = DATA_PROCESSED / "lake_satellite_observations.parquet"
    sat_status = "SKIPPED"
    if sat_path.exists():
        print("  Processing satellite features...")
        df_sat = pd.read_parquet(sat_path)
        if not df_sat.empty:
            df_sat_feat = engineer_satellite(df_sat)
            df_sat_feat.to_parquet(DATA_PROCESSED / "lake_satellite_features.parquet", index=False)
            sat_status = "SUCCESS"
        else:
            print("  [WARN] lake_satellite_observations.parquet is empty.")
    else:
        print("  [WARN] lake_satellite_observations.parquet missing.")
            
    print("  Feature engineering complete.")
    print(f"  [STATUS] Weather: {wx_status}, Satellite: {sat_status}")
    
    if wx_status == "SKIPPED" and sat_status == "SKIPPED":
        print("  [STATUS] SKIPPED")
    elif wx_status == "SUCCESS" and sat_status == "SUCCESS":
        print("  [STATUS] SUCCESS")
    elif wx_status == "SUCCESS" or sat_status == "SUCCESS":
        print("  [STATUS] PARTIAL")
    else:
        print("  [STATUS] FAILED")

if __name__ == "__main__":
    main()
