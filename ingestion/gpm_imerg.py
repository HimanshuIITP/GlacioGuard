#!/usr/bin/env python3
"""
GlacioGuard — GPM IMERG Precipitation Ingestion
Downloads and processes GPM_3IMERGDF v07 data via NASA Earthdata / earthaccess.
Extracts precipitation for target lakes.
"""

import argparse
import sys
import os
from pathlib import Path
import yaml
import pandas as pd
import geopandas as gpd
import numpy as np
from datetime import datetime, timezone

try:
    import earthaccess
except ImportError:
    earthaccess = None

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PROCESSED = BASE_DIR / "data" / "processed"
CONFIG_FILE = BASE_DIR / "config" / "test_lakes.yaml"
OBS_CONFIG = BASE_DIR / "config" / "observation_config.yaml"

def get_config():
    if not OBS_CONFIG.exists():
        return {}
    with open(OBS_CONFIG, "r") as f:
        return yaml.safe_load(f)

def get_test_lakes():
    if not CONFIG_FILE.exists():
        return []
    with open(CONFIG_FILE, "r") as f:
        return yaml.safe_load(f).get("test_lakes", [])

def check_credentials():
    if os.environ.get("EARTHDATA_TOKEN"):
        return True
    if not (os.environ.get("EARTHDATA_USERNAME") and os.environ.get("EARTHDATA_PASSWORD")) and not Path("~/.netrc").expanduser().exists():
        return False
    return True

def process_lake(lake_row, start_date, end_date, config, run_id):
    uid = lake_row["lake_uid"]
    geom = lake_row["geometry"]
    
    # 1. Geometry
    bbox = geom.bounds
    
    if not earthaccess:
        print("  [ERROR] earthaccess not installed.")
        return pd.DataFrame()
        
    try:
        earthaccess.login()
        # Search GPM_3IMERGDF v07
        results = earthaccess.search_data(
            short_name="GPM_3IMERGDF",
            version="07",
            temporal=(start_date, end_date),
            bounding_box=(bbox[0], bbox[1], bbox[2], bbox[3])
        )
    except Exception as e:
        print(f"  [ERROR] earthaccess search failed for {uid}: {e}")
        return pd.DataFrame()
        
    print(f"  Found {len(results)} potential GPM granules for {uid}")
    
    valid_observations = 0
    rejected_observations = 0
    
    records = []
    
    try:
        import xarray as xr
    except ImportError:
        xr = None
        
    for r in results:
        try:
            if not xr:
                raise ImportError("xarray is not installed.")
                
            # earthaccess.open returns a list of file-like objects
            file_objs = earthaccess.open([r])
            if not file_objs:
                continue
                
            # Use h5netcdf engine to read HDF5
            with xr.open_dataset(file_objs[0], engine="h5netcdf") as ds:
                
                # Dynamically resolve precipitation variable
                available_vars = list(ds.data_vars.keys())
                precip_var = None
                for v in ["precipitation", "precipitationCal", "precip", "HQprecipitation"]:
                    if v in available_vars:
                        precip_var = v
                        break
                        
                if not precip_var:
                    print(f"  [WARN] No precipitation variable found in {available_vars}")
                    rejected_observations += 1
                    continue
                    
                precip = ds[precip_var]
                
                # Check dimensions
                dims = list(precip.dims)
                lon_dim = "lon" if "lon" in dims else "longitude" if "longitude" in dims else None
                lat_dim = "lat" if "lat" in dims else "latitude" if "latitude" in dims else None
                
                if not lon_dim or not lat_dim:
                    print(f"  [WARN] Missing spatial dimensions in {dims}")
                    rejected_observations += 1
                    continue
                
                # Extract nearest point
                lon, lat = lake_row["longitude"], lake_row["latitude"]
                
                # Handle 0-360 longitude
                if ds[lon_dim].max() > 180 and lon < 0:
                    lon_adjusted = lon + 360
                else:
                    lon_adjusted = lon
                
                try:
                    val_array = precip.sel({lon_dim: lon_adjusted, lat_dim: lat}, method="nearest").compute()
                    val = float(val_array.item() if val_array.size == 1 else val_array.values[0])
                except Exception as e:
                    print(f"  [WARN] Spatial selection failed: {e}")
                    rejected_observations += 1
                    continue
                
                # Check for valid precipitation value
                # GPM missing values can be -9999.9. They should be >= 0.
                if np.isnan(val) or val < 0:
                    # Invalid or missing data
                    rejected_observations += 1
                    continue
                    
                dt_str = r.get("umm", {}).get("TemporalExtent", {}).get("RangeDateTime", {}).get("BeginningDateTime")
                if dt_str:
                    dt = datetime.strptime(dt_str.split(".")[0][:19], "%Y-%m-%dT%H:%M:%S")
                else:
                    dt = datetime.now(timezone.utc)
                    
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                    
                granule_id = r.get("meta", {}).get("native-id", "unknown")
                observation_id = f"{uid}_{dt.strftime('%Y%m%dT%H%M%S')}_{granule_id}"
                import hashlib
                observation_id = hashlib.sha256(observation_id.encode()).hexdigest()[:16]
                    
                records.append({
                    "observation_id": observation_id,
                    "run_id": run_id,
                    "lake_uid": uid,
                    "observation_timestamp": dt,
                    "precipitation_mm": float(val),
                    "source": "NASA Earthdata",
                    "source_product": "GPM_3IMERGDF",
                    "source_version": "07",
                    "precipitation_variable_used": precip_var,
                    "quality_flag": "good",
                    "quality_reason": "valid",
                    "processing_version": "1.0.0",
                    "retrieved_at": datetime.now(timezone.utc).isoformat()
                })
                valid_observations += 1
                
        except Exception as e:
            # Skip invalid processing, do NOT record fake rows
            print(f"  [ERROR] processing GPM granule: {e}")
            rejected_observations += 1
            continue
            
    print(f"  [METRICS] Lake {uid}: valid_observations={valid_observations}, rejected_observations={rejected_observations}")
    return pd.DataFrame(records)

def main():
    ap = argparse.ArgumentParser(description="GPM IMERG Ingestion")
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--lake-id", type=str)
    ap.add_argument("--start-date", type=str)
    ap.add_argument("--end-date", type=str)
    ap.add_argument("--run-id", type=str, default="manual")
    
    args, _ = ap.parse_known_args()
    
    print("\n" + "=" * 60)
    print("  STAGE: GPM IMERG Precipitation Ingestion")
    print("=" * 60)
    
    if not check_credentials():
        print("  [STATUS] UNAVAILABLE")
        print("  [WARN] Missing NASA Earthdata credentials. Skipping GPM.")
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
        
    all_results = []
    for idx, row in gdf.iterrows():
        print(f"  Processing GPM for {row['lake_uid']}...")
        df = process_lake(row, start, end, config, args.run_id)
        if not df.empty:
            all_results.append(df)
            
    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        raw_count = len(final_df)
        
        out_path = DATA_PROCESSED / "lake_precipitation_highfreq.parquet"
        temp_path = DATA_PROCESSED / "lake_precipitation_highfreq.tmp.parquet"
        
        if out_path.exists():
            existing_df = pd.read_parquet(out_path)
            combined = pd.concat([existing_df, final_df], ignore_index=True)
            final_df = combined.drop_duplicates(subset=["observation_id"], keep="last")
            
        final_df.to_parquet(temp_path, index=False)
        temp_path.replace(out_path)
        print(f"  Saved {len(final_df)} precipitation records to {out_path.name}")
        print("  [STATUS] SUCCESS")
    else:
        print("\nNo valid precipitation records extracted.")
        print("  [STATUS] FAILED")

if __name__ == "__main__":
    main()
