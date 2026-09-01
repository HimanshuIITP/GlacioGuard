#!/usr/bin/env python3
"""
GlacioGuard — Quality Control
Generates data coverage parquet and quality report markdown.
"""

import argparse
from pathlib import Path
import pandas as pd
from datetime import datetime, timezone

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PROCESSED = BASE_DIR / "data" / "processed"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--lake-id", type=str)
    ap.add_argument("--run-id", type=str, default="manual")
    
    args, _ = ap.parse_known_args()
    
    print("\n" + "=" * 60)
    print("  STAGE: Quality Control & Coverage")
    print("=" * 60)
    
    master_path = DATA_PROCESSED / "lake_observations.parquet"
    if not master_path.exists():
        print("  [ERROR] lake_observations.parquet not found. Cannot run Quality Control.")
        print("  [STATUS] FAILED")
        return
        
    df = pd.read_parquet(master_path)
    if df.empty:
        print("  [ERROR] Master observation table is empty.")
        print("  [STATUS] FAILED")
        return
        
    total_master_rows = len(df)
    
    # Calculate rows with valid actual observations
    has_dem = "elevation_m" in df.columns and df["elevation_m"].notna().any()
    
    metrics = {}
    current_run_id = args.run_id
    
    for src, col_actual, col_run in [
        ("WX", "actual_weather_observation", "weather_run_id"),
        ("SAT", "actual_satellite_observation", "satellite_run_id"),
        ("SNOW", "actual_snow_observation", "snow_run_id"),
        ("GPM", "actual_gpm_observation", "gpm_run_id"),
    ]:
        if col_actual in df.columns:
            valid_mask = df[col_actual] == True
            valid_total = valid_mask.sum()
            
            if col_run in df.columns:
                current = (valid_mask & (df[col_run] == current_run_id)).sum()
                cached = (valid_mask & (df[col_run] != current_run_id) & df[col_run].notna()).sum()
            else:
                current = valid_total
                cached = 0
                
            metrics[src] = {"valid": valid_total, "current": current, "cached": cached}
        else:
            metrics[src] = {"valid": 0, "current": 0, "cached": 0}
    
    # Identify stale observations
    stale_sat = df["satellite_stale"].sum() if "satellite_stale" in df.columns else 0
    stale_wx = df["weather_stale"].sum() if "weather_stale" in df.columns else 0
    stale_snow = df["snow_stale"].sum() if "snow_stale" in df.columns else 0
    stale_gpm = df["gpm_stale"].sum() if "gpm_stale" in df.columns else 0
    
    # Physical value anomalies
    anom_ndsi = 0
    if "snow_cover_fraction" in df.columns:
        anom_ndsi = ((df["snow_cover_fraction"] < 0) | (df["snow_cover_fraction"] > 1)).sum()
        
    anom_area = 0
    if "lake_area_km2" in df.columns:
        anom_area = (df["lake_area_km2"] < 0).sum()
        
    report = f"""# GlacioGuard Observation Quality Report
Generated: {datetime.now(timezone.utc).isoformat()}
Run ID: {current_run_id}

## Summary
- **Total Master Rows (Reference Grid)**: {total_master_rows}

## Valid Data Coverage (Master Cells)
- **DEM Terrain Available**: {has_dem}
- **Weather (ERA5-Land)**: {metrics['WX']['valid']} total ({metrics['WX']['current']} current, {metrics['WX']['cached']} cached) | ({(metrics['WX']['valid']/total_master_rows)*100:.1f}%)
- **Satellite (Sentinel-2)**: {metrics['SAT']['valid']} total ({metrics['SAT']['current']} current, {metrics['SAT']['cached']} cached) | ({(metrics['SAT']['valid']/total_master_rows)*100:.1f}%)
- **Snow (MODIS)**: {metrics['SNOW']['valid']} total ({metrics['SNOW']['current']} current, {metrics['SNOW']['cached']} cached) | ({(metrics['SNOW']['valid']/total_master_rows)*100:.1f}%)
- **Precipitation (GPM)**: {metrics['GPM']['valid']} total ({metrics['GPM']['current']} current, {metrics['GPM']['cached']} cached) | ({(metrics['GPM']['valid']/total_master_rows)*100:.1f}%)

## Physical Constraints & Anomalies
- **Stale Observations (Filled with NaN)**: SAT={stale_sat}, WX={stale_wx}, SNOW={stale_snow}, GPM={stale_gpm}
- **Invalid NDSI Values (<0 or >1)**: {anom_ndsi}
- **Invalid Lake Area (<0)**: {anom_area}

## Leakage Tests
See `tests/test_leakage.py` for automated temporal leakage validation.
"""

    with open(DATA_PROCESSED / "observation_quality_report.md", "w") as f:
        f.write(report)
        
    print(f"  Processed {total_master_rows} master rows.")
    print(f"  Current Valid: WX={metrics['WX']['current']}, SAT={metrics['SAT']['current']}, SNOW={metrics['SNOW']['current']}, GPM={metrics['GPM']['current']}")
    print(f"  Cached Valid:  WX={metrics['WX']['cached']}, SAT={metrics['SAT']['cached']}, SNOW={metrics['SNOW']['cached']}, GPM={metrics['GPM']['cached']}")
    
    total_time_varying = sum(m['valid'] for m in metrics.values())
    if total_time_varying == 0:
        print("  [ERROR] No valid observations obtained from any time-varying source.")
        print("  [STATUS] FAILED")
    else:
        print("  [STATUS] SUCCESS")

if __name__ == "__main__":
    main()
