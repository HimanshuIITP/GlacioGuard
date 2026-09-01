#!/usr/bin/env python3
"""
GlacioGuard — ERA5-Land Weather Ingestion
Downloads hourly temperature, precipitation, and snowfall data from the Copernicus Climate Data Store.
Extracts nearest-grid-cell values for target lakes.
"""

import argparse
import sys
import os
from pathlib import Path
import yaml
import pandas as pd
import geopandas as gpd
from datetime import datetime, timezone
from dotenv import load_dotenv
load_dotenv()
import xarray as xr
import cdsapi
import json
import math

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PROCESSED = BASE_DIR / "data" / "processed"
RAW_DIR = BASE_DIR / "data" / "raw" / "era5"
CONFIG_FILE = BASE_DIR / "config" / "test_lakes.yaml"
OBS_CONFIG = BASE_DIR / "config" / "observation_config.yaml"
SOURCES_CONFIG = BASE_DIR / "config" / "observation_sources.yaml"

def get_config():
    if not OBS_CONFIG.exists():
        return {}
    with open(OBS_CONFIG, "r") as f:
        return yaml.safe_load(f)

def get_source_meta():
    if not SOURCES_CONFIG.exists():
        return {}
    with open(SOURCES_CONFIG, "r") as f:
        return yaml.safe_load(f).get("era5_land", {})

def get_test_lakes():
    if not CONFIG_FILE.exists():
        return []
    with open(CONFIG_FILE, "r") as f:
        return yaml.safe_load(f).get("test_lakes", [])

def check_credentials():
    print(f"  [DIAGNOSTIC] sys.executable: {sys.executable}")
    print(f"  [DIAGNOSTIC] sys.prefix: {sys.prefix}")
    print(f"  [DIAGNOSTIC] sys.base_prefix: {sys.base_prefix}")
    print(f"  [DIAGNOSTIC] Path.home(): {Path.home()}")
    
    rc_path = Path.home() / ".cdsapirc"
    print(f"  [DIAGNOSTIC] {rc_path} exists: {rc_path.exists()}")
    
    has_url = bool(os.environ.get("CDSAPI_URL"))
    has_key = bool(os.environ.get("CDSAPI_KEY"))
    print(f"  [DIAGNOSTIC] CDSAPI_URL present: {has_url}")
    print(f"  [DIAGNOSTIC] CDSAPI_KEY present: {has_key}")
    
    try:
        import cdsapi
        client = cdsapi.Client(quiet=True, debug=False)
        return True
    except Exception as e:
        print(f"  [DIAGNOSTIC] cdsapi.Client() test failed: {type(e).__name__}")
        return False

def download_era5(year, month, bbox, force=False):
    """Download one month of ERA5-Land hourly data for a specific bounding box."""
    # Bbox should be [north, west, south, east]
    n, w, s, e = bbox
    # Round to 1 decimal place to ensure we cover the area and hit cached cds requests if possible
    n = math.ceil(n * 10) / 10.0
    w = math.floor(w * 10) / 10.0
    s = math.floor(s * 10) / 10.0
    e = math.ceil(e * 10) / 10.0
    
    out_file = RAW_DIR / f"era5_land_{year}_{month:02d}_{n}_{w}_{s}_{e}.nc"
    if out_file.exists() and not force:
        print(f"  [skip] {out_file.name} (cached)")
        return out_file
        
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading ERA5-Land {year}-{month:02d} for bbox [{n}, {w}, {s}, {e}]...")
    
    try:
        c = cdsapi.Client()
        c.retrieve(
            'reanalysis-era5-land',
            {
                'variable': [
                    '2m_temperature', 'total_precipitation', 'snowfall',
                ],
                'year': str(year),
                'month': f"{month:02d}",
                'day': [f"{i:02d}" for i in range(1, 32)],
                'time': [f"{i:02d}:00" for i in range(24)],
                'area': [n, w, s, e],
                'format': 'netcdf',
            },
            str(out_file)
        )
        
        import zipfile
        if zipfile.is_zipfile(out_file):
            print(f"  [INFO] Extracting NetCDF from downloaded ZIP archive...")
            extracted_path = None
            with zipfile.ZipFile(out_file, 'r') as zip_ref:
                nc_names = [name for name in zip_ref.namelist() if name.endswith('.nc')]
                if nc_names:
                    extracted_path = zip_ref.extract(nc_names[0], path=out_file.parent)
            
            if extracted_path:
                out_file.unlink()
                Path(extracted_path).rename(out_file)
                    
        return out_file
    except Exception as e:
        print(f"  [ERROR] CDS API request failed: {e}")
        if out_file.exists():
            out_file.unlink()
        return None

def process_lake(lake_row, ds_nc, run_id):
    """Extract nearest grid cell from NetCDF for a single lake."""
    uid = lake_row["lake_uid"]
    lat, lon = lake_row["latitude"], lake_row["longitude"]
    try:
        # Nearest neighbor selection
        # ERA5-Land uses latitude and longitude as coordinate names
        point_data = ds_nc.sel(latitude=lat, longitude=lon, method="nearest")
        
        df = point_data.to_dataframe().reset_index()
        
        # Convert units
        # temp: K to C
        if "t2m" in df.columns:
            df["temperature_2m_c"] = df["t2m"] - 273.15
        else:
            df["temperature_2m_c"] = pd.NA
            
        # precip: m to mm
        if "tp" in df.columns:
            df["precipitation_mm"] = df["tp"] * 1000.0
        else:
            df["precipitation_mm"] = pd.NA
            
        # snowfall: m of water equivalent to mm
        if "sf" in df.columns:
            df["snowfall_mm"] = df["sf"] * 1000.0
        else:
            df["snowfall_mm"] = pd.NA
            
        df["lake_uid"] = lake_row["lake_uid"]
        if "time" in df.columns:
            df["observation_timestamp"] = df["time"]
            df["timestamp"] = df["time"] # Keep for backward compat if needed, but we should use observation_timestamp
        elif "valid_time" in df.columns:
            df["observation_timestamp"] = df["valid_time"]
            df["timestamp"] = df["valid_time"]
            
        # Add metadata
        meta = get_source_meta()
        df["source"] = "ERA5-Land"
        df["source_product"] = meta.get("dataset_name", "Unknown")
        df["source_version"] = meta.get("version", "Unknown")
        df["retrieved_at"] = datetime.now(timezone.utc).isoformat()
        
        # Quality flags
        # Very basic check: are values finite?
        df["quality_flag"] = "good"
        df["quality_reason"] = "valid"
        
        mask_invalid = df["temperature_2m_c"].isna() | df["precipitation_mm"].isna()
        df.loc[mask_invalid, "quality_flag"] = "rejected"
        df.loc[mask_invalid, "quality_reason"] = "missing_values"
        
        df["processing_version"] = "1.0.0"
        
        df["run_id"] = run_id
        # deterministic observation_id
        df["observation_id"] = df.apply(
            lambda r: __import__('hashlib').sha256(f"{uid}_{r['observation_timestamp']}_ERA5L".encode()).hexdigest()[:16], 
            axis=1
        )
        
        cols = ["observation_id", "run_id", "lake_uid", "observation_timestamp", "temperature_2m_c", "precipitation_mm", "snowfall_mm",
                "source", "source_product", "source_version", "retrieved_at",
                "quality_flag", "quality_reason", "processing_version"]
                
        return df[[c for c in cols if c in df.columns]]
        
    except Exception as e:
        print(f"  [ERROR] processing lake {lake_row['lake_uid']}: {e}")
        return pd.DataFrame()

def main():
    ap = argparse.ArgumentParser(description="ERA5-Land Ingestion")
    ap.add_argument("--test", action="store_true", help="Run for test lakes")
    ap.add_argument("--all", action="store_true", help="Run for all lakes")
    ap.add_argument("--lake-id", type=str, help="Run for specific lake UID")
    ap.add_argument("--start-date", type=str, help="Start date (YYYY-MM-DD)")
    ap.add_argument("--end-date", type=str, help="End date (YYYY-MM-DD)")
    ap.add_argument("--force", action="store_true", help="Force redownload")
    ap.add_argument("--run-id", type=str, default="manual")
    
    args, _ = ap.parse_known_args()
    
    print("\n" + "=" * 60)
    print("  STAGE: ERA5-Land Weather Ingestion")
    print("=" * 60)
    
    if not check_credentials():
        print("  [STATUS] UNAVAILABLE")
        print("  [WARN] Missing CDS API credentials. Skipping ERA5-Land ingestion.")
        return
        
    config = get_config()
    start = args.start_date or config.get("development", {}).get("start_date", "2020-01-01")
    end = args.end_date or config.get("development", {}).get("end_date") or datetime.now().strftime("%Y-%m-%d")
    
    gj_path = DATA_PROCESSED / "india_glacial_lakes_2022.geojson"
    if not gj_path.exists():
        print(f"Error: {gj_path} not found.")
        sys.exit(1)
        
    gdf = gpd.read_file(gj_path)
    
    if args.test:
        uids = get_test_lakes()
        gdf = gdf[gdf.lake_uid.isin(uids)].copy()
    elif args.lake_id:
        gdf = gdf[gdf.lake_uid == args.lake_id].copy()
    elif not args.all:
        print("Must specify --test, --all, or --lake-id")
        sys.exit(1)
        
    if gdf.empty:
        print("No lakes selected.")
        return
        
    # Determine bounding box of all selected lakes
    bounds = gdf.total_bounds # [minx, miny, maxx, maxy] (w, s, e, n)
    bbox = [bounds[3], bounds[0], bounds[1], bounds[2]] # [n, w, s, e]
    
    # Determine months to download
    try:
        dt_start = pd.to_datetime(start)
        dt_end = pd.to_datetime(end)
        months = pd.date_range(dt_start, dt_end, freq='MS')
    except Exception as e:
        print(f"  [ERROR] Date parsing failed: {e}")
        return
        
    print(f"Extracting weather for {len(gdf)} lakes from {start} to {end}")
    
    all_results = []
    
    for dt in months:
        nc_file = download_era5(dt.year, dt.month, bbox, args.force)
        if not nc_file:
            continue
            
        print(f"  Extracting spatial data for {dt.strftime('%Y-%m')}...")
        try:
            ds = xr.open_dataset(nc_file)
            for idx, row in gdf.iterrows():
                lake_df = process_lake(row, ds, args.run_id)
                if not lake_df.empty:
                    all_results.append(lake_df)
            ds.close()
        except Exception as e:
            print(f"  [ERROR] Failed to read {nc_file.name}: {e}")
            
    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        out_path = DATA_PROCESSED / "lake_weather_observations.parquet"
        temp_path = DATA_PROCESSED / "lake_weather_observations.tmp.parquet"
        
        if out_path.exists():
            existing_df = pd.read_parquet(out_path)
            combined = pd.concat([existing_df, final_df], ignore_index=True)
            final_df = combined.drop_duplicates(subset=["observation_id"], keep="last")
            
        final_df.to_parquet(temp_path, index=False)
        temp_path.replace(out_path)
        print(f"\nSaved {len(final_df)} weather records to {out_path.name}")
        print("  [STATUS] SUCCESS")
    else:
        print("\nNo weather records extracted.")
        print("  [STATUS] FAILED")

if __name__ == "__main__":
    main()
