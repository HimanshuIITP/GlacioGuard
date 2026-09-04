#!/usr/bin/env python3
"""
GlacioGuard — Sentinel-2 Satellite Ingestion
Downloads and processes Sentinel-2 L2A data via Copernicus Data Space Ecosystem (CDSE) STAC API.
Extracts MNDWI-based water area for target lakes.
"""

import argparse
import sys
import os
from pathlib import Path
import yaml
import pandas as pd
import geopandas as gpd
from datetime import datetime, timezone
import requests
import numpy as np
import time

try:
    from pystac_client import Client
except ImportError:
    Client = None
    
try:
    import stackstac
    import xarray as xr
except ImportError:
    stackstac = None
    xr = None

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PROCESSED = BASE_DIR / "data" / "processed"
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
        return yaml.safe_load(f).get("sentinel2", {})

def get_test_lakes():
    if not CONFIG_FILE.exists():
        return []
    with open(CONFIG_FILE, "r") as f:
        return yaml.safe_load(f).get("test_lakes", [])

def get_cdse_token():
    print(f"  [DIAGNOSTIC] Authentication mode: OAuth Client Credentials")
    
    client_id = os.environ.get("CDSE_CLIENT_ID")
    client_secret = os.environ.get("CDSE_CLIENT_SECRET")
    print(f"  [DIAGNOSTIC] CDSE_CLIENT_ID present: {bool(client_id)}")
    print(f"  [DIAGNOSTIC] CDSE_CLIENT_SECRET present: {bool(client_secret)}")
    print(f"  [DIAGNOSTIC] .env file exists: {Path('.env').exists()}")
    
    if not client_id or not client_secret:
        raise ValueError("Missing CDSE_CLIENT_ID or CDSE_CLIENT_SECRET")
        
    token_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
    res = requests.post(token_url, data={
        "grant_type": "client_credentials",
        "client_id": client_id,
        "client_secret": client_secret
    })
    res.raise_for_status()
    return res.json()["access_token"]

def check_credentials():
    return bool(os.environ.get("CDSE_CLIENT_ID") and os.environ.get("CDSE_CLIENT_SECRET"))

def process_lake(lake_row, start_date, end_date, config, token, run_id):
    """Search and process Sentinel-2 data for one lake."""
    uid = lake_row["lake_uid"]
    geom = lake_row["geometry"]
    
    # 1. Configuration
    s2_config = config.get("sentinel2", {})
    buffer_m = s2_config.get("buffer_m", 250)
    cloud_max = s2_config.get("cloud_fraction_max", 0.30)
    min_water_pixels = s2_config.get("min_valid_water_pixels", 20)
    
    from processing.spatial_extraction import get_local_utm_crs
    metric_crs = get_local_utm_crs(lake_row["longitude"], lake_row["latitude"])
    
    # 2. Geometry processing
    gdf = gpd.GeoDataFrame(geometry=[geom], crs="EPSG:4326")
    buffered = gdf.to_crs(metric_crs).buffer(buffer_m)
    bounds_metric = buffered.total_bounds
    
    # Determine pixel dimensions at 10m resolution
    width = max(1, int(round((bounds_metric[2] - bounds_metric[0]) / 10.0)))
    height = max(1, int(round((bounds_metric[3] - bounds_metric[1]) / 10.0)))
    
    buffered_wgs = buffered.to_crs("EPSG:4326")
    bbox = list(buffered_wgs.total_bounds)  # [minx, miny, maxx, maxy]
    
    # 3. STAC Search via Sentinel Hub Catalog API
    stac_url = "https://sh.dataspace.copernicus.eu/api/v1/catalog/1.0.0/search"
    payload = {
        "collections": ["sentinel-2-l2a"],
        "bbox": bbox,
        "datetime": f"{start_date}T00:00:00Z/{end_date}T23:59:59Z",
        "limit": 100
    }
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            res = requests.post(stac_url, headers={"Authorization": f"Bearer {token}"}, json=payload)
            res.raise_for_status()
            items = res.json().get("features", [])
            break
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"  [ERROR] CDSE Catalog API failed: {e}")
                return pd.DataFrame()
                
    if not items:
        print(f"  No S2 scenes found for {uid}")
        return pd.DataFrame()
        
    print(f"  Found {len(items)} potential S2 scenes for {uid}")
    
    results = []
    
    # Evalscript to return B03, B11, SCL
    evalscript = """
    //VERSION=3
    function setup() {
      return {
        input: ["B03", "B11", "SCL"],
        output: { bands: 3, sampleType: "FLOAT32" }
      };
    }
    function evaluatePixel(sample) {
      return [sample.B03, sample.B11, sample.SCL];
    }
    """
    
    process_url = "https://sh.dataspace.copernicus.eu/api/v1/process"
    
    # Process each item
    for item in items:
        dt_str = item["properties"]["datetime"]
        cc = item["properties"].get("eo:cloud_cover", 100)
        scene_id = item["id"]
        
        if cc > cloud_max * 100:
            print(f"  [skip] {scene_id} - cloud cover {cc}% > max {cloud_max * 100}%")
            continue
            
        try:
            # Sentinel Hub Process API request
            proc_payload = {
                "input": {
                    "bounds": {
                        "bbox": bbox,
                        "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"}
                    },
                    "data": [
                        {
                            "type": "sentinel-2-l2a",
                            "dataFilter": {
                                "timeRange": {
                                    "from": dt_str,
                                    "to": dt_str
                                }
                            }
                        }
                    ]
                },
                "output": {
                    "width": width,
                    "height": height,
                    "responses": [
                        {
                            "identifier": "default",
                            "format": {"type": "image/tiff"}
                        }
                    ]
                },
                "evalscript": evalscript
            }
            
            res = requests.post(process_url, headers={"Authorization": f"Bearer {token}"}, json=proc_payload)
            res.raise_for_status()
            
            # Read TIFF from memory
            import rasterio
            from rasterio.io import MemoryFile
            
            with MemoryFile(res.content) as memfile:
                with memfile.open() as dataset:
                    data = dataset.read() # Shape: (3, height, width)
                    
            green = data[0]
            swir = data[1]
            scl = data[2]
            
            # Valid pixels (not clouds/shadows/nodata)
            # SCL: 3=Cloud shadows, 8=Cloud medium, 9=Cloud high, 10=Cirrus, 0=No Data
            valid_mask = ~np.isin(scl, [0, 3, 8, 9, 10])
            cloud_mask = np.isin(scl, [8, 9, 10])
            
            total_pixels = float(scl.size)
            valid_pixels = float(np.sum(valid_mask))
            valid_fraction = valid_pixels / total_pixels if total_pixels > 0 else 0.0
            cloud_fraction = float(np.sum(cloud_mask)) / total_pixels if total_pixels > 0 else 0.0
            
            if valid_pixels < min_water_pixels:
                print(f"  [skip] {scene_id} - insufficient valid pixels ({valid_pixels})")
                continue
                
            # mndwi = (Green - SWIR) / (Green + SWIR)
            denom = green + swir
            with np.errstate(divide='ignore', invalid='ignore'):
                mndwi = np.where(denom != 0, (green - swir) / denom, np.nan)
            
            # Mask to valid pixels
            mndwi_valid = mndwi[valid_mask]
            
            # Water pixels
            water_pixels = float(np.sum(mndwi_valid > 0))
            
            # Area in km2. Resolution is exactly 10m.
            pixel_area_km2 = (10 * 10) / 1_000_000.0
            lake_area_km2 = water_pixels * pixel_area_km2
            
            # Stats
            mndwi_valid = mndwi_valid[~np.isnan(mndwi_valid)]
            
            ndwi_mean = float(np.mean(mndwi_valid)) if len(mndwi_valid) > 0 else pd.NA
            ndwi_median = float(np.median(mndwi_valid)) if len(mndwi_valid) > 0 else pd.NA
            ndwi_std = float(np.std(mndwi_valid)) if len(mndwi_valid) > 0 else pd.NA
            
            # Normalize timestamp to UTC datetime
            dt = datetime.strptime(dt_str.replace("Z", "+0000"), "%Y-%m-%dT%H:%M:%S.%f%z")
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            
            import hashlib
            obs_id_str = f"{uid}_{dt.strftime('%Y%m%dT%H%M%S')}_{scene_id}"
            observation_id = hashlib.sha256(obs_id_str.encode()).hexdigest()[:16]
            
            results.append({
                "observation_id": observation_id,
                "run_id": run_id,
                "lake_uid": uid,
                "observation_timestamp": dt,
                "source": "Sentinel-2",
                "source_product": "L2A",
                "source_version": "05.00",
                "scene_id": scene_id,
                "product_id": "",
                "tile_id": "",
                "lake_area_km2": lake_area_km2,
                "water_pixel_fraction": water_pixels / valid_pixels if valid_pixels > 0 else 0.0,
                "ndwi_mean": ndwi_mean,
                "ndwi_median": ndwi_median,
                "ndwi_std": ndwi_std,
                "cloud_fraction": cloud_fraction,
                "valid_pixel_fraction": valid_fraction,
                "quality_flag": "good" if valid_fraction > 0.8 else "acceptable",
                "quality_reason": "valid",
                "processing_version": "1.0.0",
                "retrieved_at": datetime.now(timezone.utc).isoformat()
            })
            
        except Exception as e:
            # Skip invalid processing, do NOT record fake rows
            print(f"  [ERROR] processing {scene_id}: {e}")
            continue
            
    return pd.DataFrame(results)

def main():
    ap = argparse.ArgumentParser(description="Sentinel-2 Ingestion")
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--lake-id", type=str)
    ap.add_argument("--start-date", type=str)
    ap.add_argument("--end-date", type=str)
    ap.add_argument("--run-id", type=str, default="manual")
    
    args, _ = ap.parse_known_args()
    
    print("\n" + "=" * 60)
    print("  STAGE: Sentinel-2 Satellite Ingestion")
    print("=" * 60)
    
    if not check_credentials():
        print("  [STATUS] UNAVAILABLE")
        print("  [WARN] Missing Copernicus Data Space (CDSE) credentials. Skipping Sentinel-2.")
        return
        
    try:
        token = get_cdse_token()
    except Exception as e:
        print("  [STATUS] FAILED")
        print(f"  [ERROR] Failed to get CDSE token: {e}")
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
        print(f"  Processing S2 for {row['lake_uid']}...")
        df = process_lake(row, start, end, config, token, args.run_id)
        if not df.empty:
            all_results.append(df)
            
    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        raw_count = len(final_df)
        
        out_path = DATA_PROCESSED / "lake_satellite_observations.parquet"
        temp_path = DATA_PROCESSED / "lake_satellite_observations.tmp.parquet"
        
        if out_path.exists():
            existing_df = pd.read_parquet(out_path)
            combined = pd.concat([existing_df, final_df], ignore_index=True)
            final_df = combined.drop_duplicates(subset=["observation_id"], keep="last")
            
        final_df.to_parquet(temp_path, index=False)
        temp_path.replace(out_path)
        print(f"  Saved {len(final_df)} satellite records to {out_path.name}")
        print("  [STATUS] SUCCESS")
    else:
        print("\nNo valid satellite records extracted.")
        print("  [STATUS] NO_VALID_DATA")

if __name__ == "__main__":
    main()
