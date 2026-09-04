#!/usr/bin/env python3
"""
GlacioGuard — Step 2: Historical + Near-Real-Time Lake Observation Engine
End-to-End Runner
"""

import argparse
import sys
import yaml
from pathlib import Path
import os
import importlib.util
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Configuration Loader
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
CONFIG_DIR = BASE_DIR / "config"
INGESTION_DIR = BASE_DIR / "ingestion"
PROCESSING_DIR = BASE_DIR / "processing"
DATA_PROCESSED = BASE_DIR / "data" / "processed"


def load_yaml(path: Path):
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def check_access():
    """Lightweight credential validation."""
    print("=" * 60)
    print("  GlacioGuard Step 2: Credential / Access Check")
    print("=" * 60)
    
    status = {"Sentinel-2": "unavailable", "ERA5-Land": "unavailable", 
              "GPM": "unavailable", "MODIS": "unavailable", "DEM": "unavailable"}

    # 1. Sentinel-2 CDSE
    print("Checking Sentinel-2 CDSE credentials...")
    if os.environ.get("CDSE_CLIENT_ID") and os.environ.get("CDSE_CLIENT_SECRET"):
        status["Sentinel-2"] = "available"
    else:
        print("  WARN: Missing CDSE_CLIENT_ID or CDSE_CLIENT_SECRET env variables.")
        
    # 2. ERA5-Land (CDS API)
    print("Checking ERA5-Land CDS credentials...")
    if (Path.home() / ".cdsapirc").exists() or (os.environ.get("CDSAPI_URL") and os.environ.get("CDSAPI_KEY")):
        status["ERA5-Land"] = "available"
    else:
        print("  WARN: Missing ~/.cdsapirc file or CDSAPI_KEY env variables.")
        
    # 3. NASA Earthdata (GPM, MODIS)
    print("Checking NASA Earthdata credentials...")
    if (Path.home() / ".netrc").exists() or os.environ.get("EARTHDATA_TOKEN") or (os.environ.get("EARTHDATA_USERNAME") and os.environ.get("EARTHDATA_PASSWORD")):
        status["GPM"] = "available"
        status["MODIS"] = "available"
    else:
        print("  WARN: Missing ~/.netrc file, EARTHDATA_TOKEN, or EARTHDATA_USERNAME/PASSWORD env variables.")
        
    # 4. OpenTopography (DEM)
    print("Checking OpenTopography API credentials...")
    if os.environ.get("OPENTOPOGRAPHY_API_KEY"):
        status["DEM"] = "available"
    else:
        print("  WARN: Missing OPENTOPOGRAPHY_API_KEY env variable.")
        
    print("\nAccess Summary:")
    for src, state in status.items():
        print(f"  {src:12s} {state}")
    print("=" * 60)
    
    return status

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_module(module_path: Path, args_list: list):
    """Run a module as a subprocess and capture its status."""
    import subprocess
    if not module_path.exists():
        print(f"Error: Module {module_path.name} not found.")
        sys.exit(1)
        
    print(f"\n>>> Running {module_path.name} {' '.join(args_list)}")
    
    cmd = [sys.executable, str(module_path)] + args_list
    
    # Sanitize environment variables to prevent virtual environment leakage
    env = os.environ.copy()
    env.pop("PYTHONHOME", None)
    # Set PYTHONPATH to project root so submodules resolve 'processing' package
    env["PYTHONPATH"] = str(BASE_DIR)
    
    result = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=str(BASE_DIR))
    
    # Print live output
    # Filter out the annoying python C-level warning about <prefix>
    filtered_stdout = "\n".join(line for line in result.stdout.splitlines() if "Could not find platform independent libraries <prefix>" not in line)
    filtered_stderr = "\n".join(line for line in result.stderr.splitlines() if "Could not find platform independent libraries <prefix>" not in line)
    
    if filtered_stdout.strip():
        print(filtered_stdout)
    if filtered_stderr.strip():
        print(filtered_stderr, file=sys.stderr)
        
    status = "UNKNOWN"
    for line in filtered_stdout.splitlines():
        if "[STATUS]" in line:
            parts = line.split("[STATUS]")
            if len(parts) > 1:
                status = parts[1].strip()
                
    if result.returncode != 0 and status == "UNKNOWN":
        status = "FAILED"
        
    return status


def main():
    ap = argparse.ArgumentParser(description="GlacioGuard Step 2 Runner")
    ap.add_argument("--check-access", action="store_true", help="Validate credentials and exit")
    ap.add_argument("--test", action="store_true", help="Run on deterministic test lakes")
    ap.add_argument("--all", action="store_true", help="Run on all lakes")
    ap.add_argument("--lake-id", type=str, help="Run on a specific lake UID")
    ap.add_argument("--force", action="store_true", help="Force cache invalidation / re-download")
    ap.add_argument("--start-date", type=str, help="Start date (YYYY-MM-DD)")
    ap.add_argument("--end-date", type=str, help="End date (YYYY-MM-DD)")
    
    args = ap.parse_args()

    if args.check_access:
        check_access()
        sys.exit(0)
        
    if not any([args.test, args.all, args.lake_id]):
        ap.print_help()
        print("\nMust specify --test, --all, or --lake-id.")
        sys.exit(1)
        
    # Build shared args for ingestion modules
    from datetime import datetime, timezone
    current_run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    os.environ["GLACIOGUARD_RUN_ID"] = current_run_id
    
    # Resolve requested date range from args or config
    obs_cfg = load_yaml(CONFIG_DIR / "observation_config.yaml")
    req_start = args.start_date or obs_cfg.get("development", {}).get("start_date", "2020-01-01")
    req_end = args.end_date or obs_cfg.get("development", {}).get("end_date") or datetime.now().strftime("%Y-%m-%d")
    
    cmd_args = ["--run-id", current_run_id]
    if args.force:
        cmd_args.append("--force")
    if args.start_date:
        cmd_args.extend(["--start-date", args.start_date])
    if args.end_date:
        cmd_args.extend(["--end-date", args.end_date])
        
    lake_args = []
    if args.test:
        lake_args.append("--test")
    elif args.all:
        lake_args.append("--all")
    elif args.lake_id:
        lake_args.extend(["--lake-id", args.lake_id])

    print("=" * 60)
    print("  GlacioGuard Step 2: Full Observation Engine Pipeline")
    print(f"  Requested window: {req_start} to {req_end}")
    print("=" * 60)
    
    # Sequence of modules to run
    pipeline = [
        # Ingestion
        (INGESTION_DIR / "dem_download.py", lake_args + cmd_args),
        (INGESTION_DIR / "dem_terrain.py", lake_args + cmd_args),
        (INGESTION_DIR / "era5_land.py", lake_args + cmd_args),
        (INGESTION_DIR / "sentinel2_observations.py", lake_args + cmd_args),
        (INGESTION_DIR / "modis_snow.py", lake_args + cmd_args),
        (INGESTION_DIR / "gpm_imerg.py", lake_args + cmd_args),
        
        (PROCESSING_DIR / "feature_engineering.py", lake_args),
        (PROCESSING_DIR / "baselines.py", lake_args),
        (PROCESSING_DIR / "temporal_alignment.py", lake_args),
        (PROCESSING_DIR / "quality_control.py", lake_args),
        (BASE_DIR / "tests" / "test_leakage.py", lake_args)
    ]

    statuses = {}
    for mod_path, mod_args in pipeline:
        if mod_path.exists():
            st = run_module(mod_path, mod_args)
            statuses[mod_path.name] = st
        else:
            print(f"  [skip] {mod_path.name} (not implemented yet)")
            statuses[mod_path.name] = "SKIPPED"
            
    print("\n" + "=" * 60)
    print("  GlacioGuard Step 2 Final Status")
    print("=" * 60)
    for mod_name, st in statuses.items():
        print(f"  {mod_name:30s} : {st}")
        
    print_final_report(statuses, current_run_id, req_start, req_end)
    
    # Check completion condition
    dem_ok = statuses.get("dem_terrain.py") == "SUCCESS"
    time_sources = [
        statuses.get("era5_land.py"),
        statuses.get("sentinel2_observations.py"),
        statuses.get("modis_snow.py"),
        statuses.get("gpm_imerg.py")
    ]
    time_ok = any(st == "SUCCESS" for st in time_sources)
    qc_ok = statuses.get("quality_control.py") == "SUCCESS"
    leakage_ok = statuses.get("test_leakage.py") == "SUCCESS"
    
    if dem_ok and time_ok and qc_ok and leakage_ok:
        print("\n=== GlacioGuard Step 2 Complete ===")
    else:
        print("\n=== Step 2 PARTIAL / INCOMPLETE ===")
        sys.exit(1)


def print_final_report(statuses, run_id, req_start, req_end):
    import pandas as pd
    import json
    
    obs_file = DATA_PROCESSED / "lake_observations.parquet"
    if not obs_file.exists():
        return
        
    try:
        master_df = pd.read_parquet(obs_file)
        total_master = len(master_df)
    except Exception:
        total_master = 0
        
    # Requested date range boundaries for filtering
    dt_start = pd.to_datetime(req_start).tz_localize("UTC")
    dt_end = pd.to_datetime(req_end).tz_localize("UTC") + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        
    # Analyze processed files -- filter by requested date range
    metrics = {}
    sources = [
        ("ERA5-Land", "era5_land.py", "lake_weather_observations.parquet", "observation_timestamp"),
        ("Sentinel-2", "sentinel2_observations.py", "lake_satellite_observations.parquet", "observation_timestamp"),
        ("MODIS", "modis_snow.py", "lake_snow_observations.parquet", "observation_timestamp"),
        ("GPM IMERG", "gpm_imerg.py", "lake_precipitation_highfreq.parquet", "observation_timestamp")
    ]
    
    for name, script, file_name, ts_col in sources:
        raw_path = DATA_PROCESSED / file_name
        current = 0
        cached = 0
        total_in_file = 0
        in_range = 0
        if raw_path.exists():
            try:
                df = pd.read_parquet(raw_path)
                total_in_file = len(df)
                
                # total_in_file is Raw/cache rows
                if ts_col in df.columns:
                    df[ts_col] = pd.to_datetime(df[ts_col], utc=True)
                    # Filter df to only in-range observations for subsequent counts
                    df = df[
                        (df[ts_col] >= dt_start) &
                        (df[ts_col] <= dt_end)
                    ]
                in_range = len(df)
                
                # Now calculate Current Valid and Cached Valid only on the in-range observations
                if "run_id" in df.columns:
                    current = (df["run_id"] == run_id).sum()
                    cached = (df["run_id"] != run_id).sum()
                else:
                    cached = len(df)
            except Exception:
                pass
        
        metrics[name] = {
            "status": statuses.get(script, "UNKNOWN"),
            "current": int(current),
            "cached": int(cached),
            "total_in_file": int(total_in_file),
            "in_range": int(in_range)
        }
    
    print(f"\nRequested processing window: {req_start} to {req_end}")
    print(f"\nSource      | Raw/cache | In-range | Current Valid | Cached Valid | Status     ")
    print("-" * 85)
    for name, m in metrics.items():
        print(f"{name:11s} | {m['total_in_file']:<9} | {m['in_range']:<8} | {m['current']:<13} | {m['cached']:<12} | {m['status']:10}")
        
    print("\nDEM")
    print(f"Acquisition: {statuses.get('dem_download.py', 'UNAVAILABLE')}")
    print(f"Extraction: {statuses.get('dem_terrain.py', 'UNKNOWN')}")
    print("Data: AVAILABLE (static, not date-filtered)")
    
    # Baselines: report period separately
    baselines_path = DATA_PROCESSED / "lake_baselines.parquet"
    if baselines_path.exists():
        try:
            df_b = pd.read_parquet(baselines_path)
            if not df_b.empty and "baseline_period_start" in df_b.columns:
                bp_start = df_b["baseline_period_start"].dropna().min()
                bp_end = df_b["baseline_period_end"].dropna().max()
                print(f"\nBaseline period: {bp_start} to {bp_end} (separate from execution window {req_start} to {req_end})")
                print(f"Baseline records: {len(df_b)}")
        except Exception:
            pass
    
    print(f"\nFinal Metrics:")
    print(f"- Master reference rows: {total_master}")
    
    # Save run manifest
    manifest_dir = DATA_PROCESSED / "runs" / run_id
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "run_id": run_id,
        "requested_start": req_start,
        "requested_end": req_end,
        "metrics": metrics,
        "statuses": statuses,
        "master_reference_rows": int(total_master)
    }
    with open(manifest_dir / "run_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

if __name__ == "__main__":
    main()
