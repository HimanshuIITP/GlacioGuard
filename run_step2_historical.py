#!/usr/bin/env python3
import sys
import yaml
import subprocess
import argparse
from pathlib import Path
import pandas as pd
from datetime import datetime, timezone

BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
CONFIG_PATH = BASE_DIR / "config" / "historical_observation_config.yaml"

def run_script(script_path, args):
    cmd = [sys.executable, str(script_path)] + args
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [ERROR] {script_path.name} failed with exit code {result.returncode}")
        print(result.stderr)
        return False
    return True

def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

def get_existing_coverage(lake_uid, source_name):
    """Returns a set of 'YYYY-MM-DD' dates already covered for this lake in the processed datasets."""
    # Map source config name to parquet files and timestamp columns
    mapping = {
        "era5_land": ("lake_weather_observations.parquet", "observation_timestamp"),
        "modis_snow": ("lake_snow_observations.parquet", "observation_timestamp"),
        "sentinel2": ("lake_satellite_observations.parquet", "observation_timestamp"),
        "gpm_imerg": ("lake_weather_observations.parquet", "observation_timestamp") # sometimes weather
    }
    
    if source_name not in mapping:
        return set()
        
    file_name, ts_col = mapping[source_name]
    file_path = PROCESSED_DIR / file_name
    
    if not file_path.exists():
        return set()
        
    df = pd.read_parquet(file_path)
    if df.empty or lake_uid not in df["lake_uid"].values:
        return set()
        
    sub = df[df["lake_uid"] == lake_uid]
    if ts_col not in sub.columns:
        return set()
        
    dates = pd.to_datetime(sub[ts_col]).dt.strftime("%Y-%m-%d").unique()
    return set(dates)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("  GlacioGuard Step 2.5 — Incremental Historical Acquisition")
    print("=" * 60)
    
    config = load_config()
    
    events_path = PROCESSED_DIR / "glof_events.parquet"
    matches_path = PROCESSED_DIR / "glof_event_lake_matches.parquet"
    if not events_path.exists() or not matches_path.exists():
        print("Missing events or matches.")
        sys.exit(1)
        
    df_events = pd.read_parquet(events_path)
    df_matches = pd.read_parquet(matches_path)
    
    # Positive Eligibility: HIGH_CONFIDENCE_MATCH + (HIGH or MEDIUM event conf)
    df_full = df_events.merge(df_matches, on="event_id", how="inner")
    mask_eligible = (df_full["match_status"] == "HIGH_CONFIDENCE_MATCH") & (df_full["event_confidence_x"].isin(["HIGH", "MEDIUM"]))
    pos_events = df_full[mask_eligible].copy()
    
    if pos_events.empty:
        print("No eligible positive events found.")
        sys.exit(0)
        
    pre_days = pd.Timedelta(days=45) # Hardcoded 45 days as requested by user
    post_days = pd.Timedelta(days=45)
    
    ingestion_scripts = {
        "era5_land": BASE_DIR / "ingestion" / "era5_land.py",
        "modis_snow": BASE_DIR / "ingestion" / "modis_snow.py",
        "gpm_imerg": BASE_DIR / "ingestion" / "gpm_imerg.py",
        "sentinel2": BASE_DIR / "ingestion" / "sentinel2_observations.py"
    }
    
    stats = {
        "cache_hits": 0,
        "cache_misses": 0,
        "requested_intervals": 0,
        "downloaded_intervals": 0,
        "skipped_existing_intervals": 0
    }
    
    for _, evt in pos_events.iterrows():
        lake_uid = evt["lake_uid"]
        
        date_str = str(evt["event_date"])
        if len(date_str) == 10:
            event_dt = pd.to_datetime(date_str + " 00:00:00", utc=True)
        else:
            event_dt = pd.to_datetime(date_str, utc=True)
            
        start_dt = event_dt - pre_days
        end_dt = event_dt + post_days
        
        print(f"\n--- Checking historical window for {lake_uid} (Event: {event_dt.strftime('%Y-%m-%d')}) ---")
        
        for src_name, src_conf in config["sources"].items():
            if not src_conf.get("enabled", False):
                continue
                
            earliest = pd.to_datetime(src_conf["earliest_date"]).tz_localize('UTC')
            if end_dt < earliest:
                print(f"  [SKIP] {src_name} - sensor not available until {earliest.strftime('%Y-%m-%d')}")
                continue
                
            adj_start = max(start_dt, earliest)
            
            # Check existing dates
            existing_dates = get_existing_coverage(lake_uid, src_name) if not args.force else set()
            
            required_dates = pd.date_range(adj_start, end_dt, freq="D").strftime("%Y-%m-%d").tolist()
            missing_dates = [d for d in required_dates if d not in existing_dates]
            
            stats["requested_intervals"] += 1
            
            if not missing_dates:
                print(f"  [CACHE HIT] {src_name} fully covered ({len(required_dates)} days). Skipping.")
                stats["cache_hits"] += 1
                stats["skipped_existing_intervals"] += 1
                continue
                
            stats["cache_misses"] += 1
            stats["downloaded_intervals"] += 1
            
            # We fetch from min missing to max missing (simplest interval encompassing missing)
            fetch_start = min(missing_dates)
            fetch_end = max(missing_dates)
            
            print(f"  [CACHE MISS] {src_name} missing {len(missing_dates)}/{len(required_dates)} days. Fetching {fetch_start} to {fetch_end}...")
            
            script_path = ingestion_scripts.get(src_name)
            if script_path and script_path.exists():
                cmd_args = ["--lake-id", lake_uid, "--start-date", fetch_start, "--end-date", fetch_end]
                if args.force:
                    cmd_args.append("--force")
                run_script(script_path, cmd_args)
            elif src_name != "dem":
                print(f"  [WARN] Ingestion script for {src_name} not found.")
                
    print("\n" + "-" * 40)
    print("Historical Download Audit")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print("-" * 40)
        
    print("\n>>> Rebuilding Historical Alignment")
    run_script(BASE_DIR / "processing" / "build_historical_observation_archive.py", ["--force"] if args.force else [])
    print("  [STATUS] SUCCESS")

if __name__ == "__main__":
    main()