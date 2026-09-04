#!/usr/bin/env python3
import sys
import yaml
import subprocess
import argparse
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
CONFIG_PATH = BASE_DIR / "config" / "historical_observation_config.yaml"

def run_script(script_path, args):
    print(f"\n>>> Running {script_path.name} {' '.join(args)}")
    cmd = [sys.executable, str(script_path)] + args
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"  [ERROR] {script_path.name} failed with exit code {result.returncode}")
        sys.exit(result.returncode)

def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("  GlacioGuard Step 2.5 — Historical Observation Acquisition")
    print("=" * 60)
    
    config = load_config()
    
    events_path = PROCESSED_DIR / "glof_label_eligible_events.parquet"
    if not events_path.exists():
        print("Missing eligible events.")
        sys.exit(1)
        
    df_events = pd.read_parquet(events_path)
    pos_events = df_events[df_events["eligibility_status"] == "ELIGIBLE_POSITIVE"]
    
    if pos_events.empty:
        print("No eligible positive events found.")
        sys.exit(1)
        
    pre_days = pd.Timedelta(days=config["event_windows"]["pre_event_days"])
    post_days = pd.Timedelta(days=config["event_windows"]["post_event_days"])
    
    # We map the ingestion script names here
    ingestion_scripts = {
        "era5_land": BASE_DIR / "ingestion" / "era5_land.py",
        "modis_snow": BASE_DIR / "ingestion" / "modis_snow.py",
        "gpm_imerg": BASE_DIR / "ingestion" / "gpm_imerg.py",
        "sentinel2": BASE_DIR / "ingestion" / "sentinel2_observations.py"
    }
    
    for _, evt in pos_events.iterrows():
        lake_uid = evt["lake_uid"]
        event_dt = pd.to_datetime(evt["event_time_parsed"])
        
        start_date = (event_dt - pre_days).strftime("%Y-%m-%d")
        end_date = (event_dt + post_days).strftime("%Y-%m-%d")
        
        print(f"\n--- Processing historical window for {lake_uid} (Event: {event_dt.strftime('%Y-%m-%d')}) ---")
        print(f"Window: {start_date} to {end_date}")
        
        for src_name, src_conf in config["sources"].items():
            if not src_conf.get("enabled", False):
                continue
                
            earliest = pd.to_datetime(src_conf["earliest_date"])
            if event_dt + post_days < earliest:
                print(f"  [SKIP] {src_name} - sensor not available until {earliest.strftime('%Y-%m-%d')}")
                continue
                
            script_path = ingestion_scripts.get(src_name)
            if script_path and script_path.exists():
                # We can't fetch if the start date is before earliest, we adjust start_date for that source
                adj_start = max(pd.to_datetime(start_date), earliest).strftime("%Y-%m-%d")
                
                cmd_args = ["--lake-id", lake_uid, "--start-date", adj_start, "--end-date", end_date]
                if args.force:
                    cmd_args.append("--force")
                run_script(script_path, cmd_args)
            elif src_name != "dem":
                print(f"  [WARN] Ingestion script for {src_name} not found.")
                
    # Now run the historical alignment
    run_script(BASE_DIR / "processing" / "build_historical_observation_archive.py", ["--force"] if args.force else [])

if __name__ == "__main__":
    main()
