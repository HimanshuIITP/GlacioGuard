#!/usr/bin/env python3
"""
GlacioGuard — MODIS Snow Cover Ingestion
Downloads and processes MOD10A1 v061 data via NASA Earthdata / earthaccess.
Extracts snow cover fraction for target lakes.
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
    
    # 1. Config
    buffer_m = config.get("snow", {}).get("buffer_m", 250)
    
    from processing.spatial_extraction import get_local_utm_crs
    metric_crs = get_local_utm_crs(lake_row["longitude"], lake_row["latitude"])
    
    # 2. Geometry
    gdf = gpd.GeoDataFrame(geometry=[geom], crs="EPSG:4326")
    buffered = gdf.to_crs(metric_crs).buffer(buffer_m)
    buffered_wgs = buffered.to_crs("EPSG:4326")
    bbox = list(buffered_wgs.total_bounds)
    
    if not earthaccess:
        print("  [ERROR] earthaccess not installed.")
        return pd.DataFrame()
        
    try:
        earthaccess.login()
        # Search MOD10A1 v61
        results = earthaccess.search_data(
            short_name="MOD10A1",
            version="61",
            temporal=(start_date, end_date),
            bounding_box=(bbox[0], bbox[1], bbox[2], bbox[3])
        )
    except Exception as e:
        print(f"  [ERROR] earthaccess search failed for {uid}: {e}")
        return pd.DataFrame()
        
    if not results:
        return pd.DataFrame()
        
    print(f"  Found {len(results)} potential MODIS granules for {uid}")
    
    records = []
    
    try:
        import xarray as xr
        import pyproj
        import re
    except ImportError:
        xr = None
        
    if not xr:
        print("  [ERROR] xarray or pyproj not installed.")
        return pd.DataFrame()
        
    # Set up Sinusoidal projection
    sinu_proj = pyproj.Proj("+proj=sinu +R=6371007.181 +nadgrids=@null +wktext")
    wgs84 = pyproj.Proj("EPSG:4326")
    transformer = pyproj.Transformer.from_proj(wgs84, sinu_proj, always_xy=True)
    
    # Project lake bounds corners to Sinusoidal
    corners = [
        (bbox[0], bbox[1]), (bbox[2], bbox[1]),
        (bbox[0], bbox[3]), (bbox[2], bbox[3])
    ]
    proj_corners = [transformer.transform(x, y) for x, y in corners]
    lake_min_x = min(x for x, y in proj_corners)
    lake_max_x = max(x for x, y in proj_corners)
    lake_min_y = min(y for x, y in proj_corners)
    lake_max_y = max(y for x, y in proj_corners)
        
    for r in results:
        # Download granule locally
        try:
            paths = earthaccess.download(r, local_path=str(DATA_PROCESSED / "temp"))
            if not paths:
                continue
            filepath = paths[0]
            
            with xr.open_dataset(filepath, engine="netcdf4", mask_and_scale=False) as ds:
                if "NDSI_Snow_Cover" not in ds.variables:
                    continue
                    
                meta = ds.attrs.get("StructMetadata.0", "")
                
                # Parse metadata
                ul_match = re.search(r"UpperLeftPointMtrs=\(([^,]+),([^)]+)\)", meta)
                lr_match = re.search(r"LowerRightPointMtrs=\(([^,]+),([^)]+)\)", meta) or re.search(r"LowerRightMtrs=\(([^,]+),([^)]+)\)", meta)
                xdim_match = re.search(r"XDim=(\d+)", meta)
                ydim_match = re.search(r"YDim=(\d+)", meta)
                
                if not (ul_match and lr_match and xdim_match and ydim_match):
                    continue
                    
                ul_x, ul_y = float(ul_match.group(1)), float(ul_match.group(2))
                lr_x, lr_y = float(lr_match.group(1)), float(lr_match.group(2))
                xdim, ydim = int(xdim_match.group(1)), int(ydim_match.group(1))
                
                # Compute indices
                col_start = int((lake_min_x - ul_x) / (lr_x - ul_x) * xdim)
                col_end = int((lake_max_x - ul_x) / (lr_x - ul_x) * xdim) + 1
                row_start = int((ul_y - lake_max_y) / (ul_y - lr_y) * ydim)
                row_end = int((ul_y - lake_min_y) / (ul_y - lr_y) * ydim) + 1
                
                # Clamp indices
                col_start = max(0, min(col_start, xdim - 1))
                col_end = max(0, min(col_end, xdim))
                row_start = max(0, min(row_start, ydim - 1))
                row_end = max(0, min(row_end, ydim))
                
                if col_start >= col_end or row_start >= row_end:
                    # Bounding box is outside this granule
                    continue
                    
                # Subset data using dynamic dimension names
                ndsi_var = ds["NDSI_Snow_Cover"]
                dims = ndsi_var.dims
                y_dim = next((d for d in dims if "YDim" in d), "YDim")
                x_dim = next((d for d in dims if "XDim" in d), "XDim")
                
                snow_data = ndsi_var.isel({x_dim: slice(col_start, col_end), y_dim: slice(row_start, row_end)})
                
                # Apply quality masking BEFORE computing final snow fraction
                # valid range = 0-100, cloud = 250, fill/night = 211,237,239,255
                raw = snow_data.values
                subset_pixels = raw.size
                cloud_pixels = int(np.sum(raw == 250))
                
                valid_mask = (raw >= 0) & (raw <= 100)
                valid_pixels = int(np.sum(valid_mask))
                
                if valid_pixels == 0:
                    print(f"  [INFO] Granule rejected: 0/{subset_pixels} valid pixels (cloud={cloud_pixels})")
                    continue
                    
                snow_pixels = int(np.sum((raw > 0) & valid_mask))
                snow_fraction = float(snow_pixels) / float(valid_pixels)
                
                # Diagnostic output for the first processed granule
                if not records:
                    print(f"  [MODIS DIAGNOSTIC]")
                    print(f"    Variable: NDSI_Snow_Cover")
                    print(f"    Dims: {dims}")
                    print(f"    Subset pixels: {subset_pixels}")
                    print(f"    Valid pixels: {valid_pixels}")
                    print(f"    Cloud pixels: {cloud_pixels}")
                    print(f"    Snow pixels: {snow_pixels}")
                    print(f"    Fraction: {snow_fraction:.4f}")

                
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
                    "snow_cover_fraction": snow_fraction,
                    "source": "NASA Earthdata",
                    "source_product": "MOD10A1",
                    "source_version": "61",
                    "quality_flag": "good",
                    "quality_reason": "valid",
                    "processing_version": "1.0.0",
                    "retrieved_at": datetime.now(timezone.utc).isoformat()
                })
                
            # Cleanup
            if os.path.exists(filepath):
                os.remove(filepath)
                
        except Exception as e:
            print(f"  [ERROR] processing MODIS granule: {e}")
            continue
            
    return pd.DataFrame(records)

def main():
    ap = argparse.ArgumentParser(description="MODIS Snow Ingestion")
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--lake-id", type=str)
    ap.add_argument("--start-date", type=str)
    ap.add_argument("--end-date", type=str)
    ap.add_argument("--run-id", type=str, default="manual")
    
    args, _ = ap.parse_known_args()
    
    print("\n" + "=" * 60)
    print("  STAGE: MODIS Snow Ingestion")
    print("=" * 60)
    
    if not check_credentials():
        print("  [STATUS] UNAVAILABLE")
        print("  [WARN] Missing NASA Earthdata credentials. Skipping MODIS.")
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
        print(f"  Processing MODIS for {row['lake_uid']}...")
        df = process_lake(row, start, end, config, args.run_id)
        if df is not None and not df.empty:
            all_results.append(df)
            
    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        out_path = DATA_PROCESSED / "lake_snow_observations.parquet"
        temp_path = DATA_PROCESSED / "lake_snow_observations.tmp.parquet"
        
        if out_path.exists():
            existing_df = pd.read_parquet(out_path)
            combined = pd.concat([existing_df, final_df], ignore_index=True)
            # Deduplicate by observation_id, keeping the latest (which was appended last)
            final_df = combined.drop_duplicates(subset=["observation_id"], keep="last")
            
        final_df.to_parquet(temp_path, index=False)
        temp_path.replace(out_path) # Atomic replace
        print(f"\nSaved {len(final_df)} snow records to {out_path.name}")
        print("  [STATUS] SUCCESS")
    else:
        print("\nNo valid snow records extracted.")
        print("  [STATUS] FAILED")

if __name__ == "__main__":
    main()
