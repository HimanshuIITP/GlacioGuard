#!/usr/bin/env python3
"""
GlacioGuard — DEM Tile Downloader
Determines required SRTM 30m (SRTMGL1) tiles for test lakes and caches them locally.
Uses the OpenTopography Global DEM API.
"""

import argparse
import sys
import os
import requests
import geopandas as gpd
from pathlib import Path
import yaml
import math

# We must import from our processing directory
from processing.spatial_extraction import get_local_utm_crs, get_dem_tile_name

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PROCESSED = BASE_DIR / "data" / "processed"
DEM_RAW_DIR = BASE_DIR / "data" / "raw" / "dem"
CONFIG_FILE = BASE_DIR / "config" / "test_lakes.yaml"

OT_API_URL = "https://portal.opentopography.org/API/globaldem"

def get_test_lakes():
    if not CONFIG_FILE.exists():
        print(f"Error: {CONFIG_FILE} not found.")
        sys.exit(1)
    with open(CONFIG_FILE, "r") as f:
        conf = yaml.safe_load(f)
    return conf.get("test_lakes", [])

def get_required_tiles(lakes_gdf, buffer_m=250):
    """Get unique 1x1 degree DEM tiles required for the selected lakes."""
    unique_tiles = set()
    
    for idx, row in lakes_gdf.iterrows():
        geom_wgs = row.geometry
        # Project to metric CRS for buffering, then back to WGS84
        centroid = geom_wgs.centroid
        metric_crs = get_local_utm_crs(centroid.x, centroid.y)
        
        lake_m = gpd.GeoSeries([geom_wgs], crs="EPSG:4326").to_crs(metric_crs)
        buffered = lake_m.buffer(buffer_m)
        buffered_wgs = buffered.to_crs("EPSG:4326")
        
        bounds = buffered_wgs.total_bounds  # (minx, miny, maxx, maxy)
        
        # A 1x1 degree application cache region covering this bounds
        min_lon, min_lat, max_lon, max_lat = bounds
        
        # Floor/ceil to integers to get 1x1 degree grid cells
        s = math.floor(min_lat)
        n = math.ceil(max_lat)
        w = math.floor(min_lon)
        e = math.ceil(max_lon)
        
        # A single lake might cross a 1-degree boundary
        for lat in range(s, n):
            for lon in range(w, e):
                unique_tiles.add((lat, lon))
                
    return unique_tiles

def download_dem_tile(lat, lon, api_key, force=False):
    """Download 1x1 degree DEM tile."""
    s = lat
    n = lat + 1
    w = lon
    e = lon + 1
    
    # OpenTopography expects a slightly larger bbox to avoid edge artifacts 
    # but 1x1 integer bounds are standard for SRTM.
    tile_name = get_dem_tile_name(lat, lon)
    out_path = DEM_RAW_DIR / tile_name
    
    if out_path.exists() and not force:
        print(f"  [skip] {tile_name} (cached)")
        return out_path
        
    print(f"  Downloading {tile_name}...")
    DEM_RAW_DIR.mkdir(parents=True, exist_ok=True)
    
    params = {
        "demtype": "SRTMGL1",
        "south": s,
        "north": n,
        "west": w,
        "east": e,
        "outputFormat": "GTiff",
        "API_Key": api_key
    }
    
    try:
        r = requests.get(OT_API_URL, params=params, stream=True)
        if r.status_code == 401:
            print("  [ERROR] Invalid OpenTopography API Key.")
            return None
        r.raise_for_status()
        
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                
        print(f"  OK   {tile_name} ({out_path.stat().st_size / 1024:.1f} KB)")
        return out_path
    except Exception as e:
        print(f"  [ERROR] Download failed: {e}")
        if out_path.exists():
            out_path.unlink()
        return None

def main():
    ap = argparse.ArgumentParser(description="DEM Downloader")
    ap.add_argument("--test", action="store_true", help="Run for test lakes")
    ap.add_argument("--all", action="store_true", help="Run for all lakes")
    ap.add_argument("--lake-id", type=str, help="Run for specific lake UID")
    ap.add_argument("--force", action="store_true", help="Force re-download")
    
    # Ignore extra args passed by runner
    args, _ = ap.parse_known_args()
    
    print("\n" + "=" * 60)
    print("  STAGE: DEM Tile Downloader")
    print("=" * 60)
    
    if not os.environ.get("OPENTOPOGRAPHY_API_KEY"):
        print("  [STATUS] UNAVAILABLE")
        print("  [WARN] OPENTOPOGRAPHY_API_KEY not found in environment. Skipping DEM download.")
        return
        
    csv_path = DATA_PROCESSED / "india_glacial_lakes_2022.csv"
    if not csv_path.exists():
        print(f"Error: {csv_path} not found. Run Step 1 first.")
        sys.exit(1)
        
    # Read inventory and keep geometries for buffering
    # Wait, the CSV has geometry as WKT, so we should read the GeoJSON
    gj_path = DATA_PROCESSED / "india_glacial_lakes_2022.geojson"
    print(f"Loading {gj_path.name}...")
    gdf = gpd.read_file(gj_path)
    
    if args.test:
        uids = get_test_lakes()
        gdf = gdf[gdf.lake_uid.isin(uids)].copy()
        print(f"Test mode: processing {len(gdf)} lakes")
    elif args.lake_id:
        gdf = gdf[gdf.lake_uid == args.lake_id].copy()
        print(f"Single lake mode: processing {len(gdf)} lake")
    elif args.all:
        print(f"Full mode: processing {len(gdf)} lakes")
    else:
        print("Must specify --test, --all, or --lake-id")
        sys.exit(1)
        
    if gdf.empty:
        print("No lakes selected.")
        sys.exit(0)
        
    print("Calculating required DEM 1x1 degree tiles...")
    required_tiles = get_required_tiles(gdf)
    
    success_count = 0
    for (lat, lon) in required_tiles:
        res = download_dem_tile(lat, lon, os.environ.get("OPENTOPOGRAPHY_API_KEY"), args.force)
        if res:
            success_count += 1
            
    print(f"\nDEM cache updated. {success_count}/{len(required_tiles)} tiles ready.")
    
    if success_count == len(required_tiles) and len(required_tiles) > 0:
        print("  [STATUS] SUCCESS")
    elif success_count > 0:
        print("  [STATUS] PARTIAL")
    else:
        print("  [STATUS] FAILED")
    
if __name__ == "__main__":
    main()
