#!/usr/bin/env python3
"""
GlacioGuard — DEM Terrain Extraction
Extracts static terrain features (elevation, slope, aspect) for lakes using cached SRTM tiles.
"""

import argparse
import sys
import os
import yaml
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
import math
import rasterio
import rasterio.mask
from processing.spatial_extraction import get_local_utm_crs, circular_mean, get_dem_tile_name
from rasterio.merge import merge

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PROCESSED = BASE_DIR / "data" / "processed"
DEM_RAW_DIR = BASE_DIR / "data" / "raw" / "dem"
CONFIG_FILE = BASE_DIR / "config" / "test_lakes.yaml"
OBS_CONFIG = BASE_DIR / "config" / "observation_config.yaml"

def get_config():
    if not OBS_CONFIG.exists():
        return {"terrain": {"buffer_m": 250}}
    with open(OBS_CONFIG, "r") as f:
        return yaml.safe_load(f)

def get_test_lakes():
    if not CONFIG_FILE.exists():
        print(f"Error: {CONFIG_FILE} not found.")
        sys.exit(1)
    with open(CONFIG_FILE, "r") as f:
        conf = yaml.safe_load(f)
    return conf.get("test_lakes", [])

def calculate_slope_aspect(elevation, resolution_m):
    """Calculate slope and aspect using metric gradient."""
    # resolution_m is (dx, dy) in meters.
    dx_m = resolution_m[0]
    dy_m = resolution_m[1]
    
    # Gradients
    gy, gx = np.gradient(elevation, dy_m, dx_m)
    
    # Slope
    slope = np.degrees(np.arctan(np.sqrt(gx**2 + gy**2)))
    
    # Aspect
    aspect = np.degrees(np.arctan2(-gy, gx))
    aspect = np.where(aspect < 0, 90 - aspect, 360 - aspect + 90)
    aspect = np.where(aspect >= 360, aspect - 360, aspect)
    
    return slope, aspect

def process_lake(lake_uid, geom_wgs, buffer_m, cached_tiles, run_id):
    """Extract terrain statistics for one lake."""
    # Determine correct local UTM CRS
    centroid = geom_wgs.centroid
    metric_crs = get_local_utm_crs(centroid.x, centroid.y)
    
    # Buffer the geometry in the correct metric CRS, then transform back to WGS84
    gdf = gpd.GeoDataFrame(geometry=[geom_wgs], crs="EPSG:4326")
    buffered = gdf.to_crs(metric_crs).buffer(buffer_m)
    buffered_wgs = buffered.to_crs("EPSG:4326")
    mask_geom_wgs = [buffered_wgs.iloc[0]]
    
    bounds = buffered_wgs.total_bounds
    min_lon, min_lat, max_lon, max_lat = bounds
    
    s = math.floor(min_lat)
    n = math.ceil(max_lat)
    w = math.floor(min_lon)
    e = math.ceil(max_lon)
    
    # Collect all intersecting tiles
    src_files_to_mosaic = []
    try:
        for lat in range(s, n):
            for lon in range(w, e):
                tile_name = get_dem_tile_name(lat, lon)
                tile_path = DEM_RAW_DIR / tile_name
                if tile_path.exists():
                    src_files_to_mosaic.append(rasterio.open(tile_path))
                else:
                    print(f"  [WARN] Missing DEM tile {tile_name} for lake {lake_uid}. Run dem_download.py.")
        
        if not src_files_to_mosaic:
            return None
            
        # Merge tiles
        mosaic, out_trans = merge(src_files_to_mosaic, bounds=tuple(bounds))
        
        # We need to write the mosaic to a MemoryFile to use rasterio.mask, 
        # or we can mask it directly if we construct an affine transform.
        # However, rasterio.mask.mask expects a dataset object.
        from rasterio.io import MemoryFile
        
        profile = src_files_to_mosaic[0].profile.copy()
        profile.update({
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": out_trans
        })
        
        with MemoryFile() as memfile:
            with memfile.open(**profile) as mem:
                mem.write(mosaic)
                out_image, out_transform = rasterio.mask.mask(mem, mask_geom_wgs, crop=True)
                out_meta = mem.meta
            
            # Since we need accurate slope, we must reproject this small window to the metric CRS
            from rasterio.warp import calculate_default_transform, reproject, Resampling
            
            dst_crs = metric_crs
            # Calculate transform for metric projection
            src_crs = src_files_to_mosaic[0].crs
            dst_transform, width, height = calculate_default_transform(
                src_crs, dst_crs, out_image.shape[2], out_image.shape[1],
                *rasterio.transform.array_bounds(out_image.shape[1], out_image.shape[2], out_transform)
            )
            
            dst_image = np.zeros((1, height, width), dtype=out_image.dtype)
            
            reproject(
                source=out_image,
                destination=dst_image,
                src_transform=out_transform,
                src_crs=src_crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=Resampling.bilinear
            )
            
        elev = dst_image[0]
        # Mask out nodata
        valid = (elev != out_meta['nodata']) & (elev > -500) # Basic validity check
        if not valid.any():
            return None
            
        elev_valid = elev[valid]
        mean_elev = elev_valid.mean()
        
        # Calculate slope/aspect in meters using the reprojected transform scale
        resolution_m = (abs(dst_transform.a), abs(dst_transform.e))
        slope, aspect = calculate_slope_aspect(elev, resolution_m)
        
        # Now mask perfectly to the buffered polygon
        # (Though technically it's already masked, reprojection might add artifacts at edges)
        mean_slope = slope[valid].mean()
        max_slope = slope[valid].max()
        mean_aspect = circular_mean(aspect[valid])
        
        # Close open source files
        for src in src_files_to_mosaic:
            src.close()
            
        import hashlib
        obs_id_str = f"{lake_uid}_OpenTopography_SRTMGL1"
        observation_id = hashlib.sha256(obs_id_str.encode()).hexdigest()[:16]
        
        return {
            "observation_id": observation_id,
            "run_id": run_id,
            "lake_uid": lake_uid,
            "elevation_m": float(mean_elev),
            "mean_slope_deg": float(mean_slope),
            "max_slope_deg": float(max_slope),
            "mean_aspect_deg": float(mean_aspect),
            "dem_source": "OpenTopography",
            "dem_product": "SRTMGL1",
            "dem_version": "v3",
            "processing_version": "1.0.0",
            "quality_flag": "good",
            "quality_reason": "valid"
        }

        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"  [ERROR] processing {lake_uid}: {e}")
        for src in src_files_to_mosaic:
            src.close()
        return None

def main():
    ap = argparse.ArgumentParser(description="DEM Terrain Extraction")
    ap.add_argument("--test", action="store_true", help="Run for test lakes")
    ap.add_argument("--all", action="store_true", help="Run for all lakes")
    ap.add_argument("--lake-id", type=str, help="Run for specific lake UID")
    ap.add_argument("--force", action="store_true", help="Force rebuild (ignored for processing, always rebuilds)")
    ap.add_argument("--run-id", type=str, default="manual")
    
    # Ignore extra args passed by runner
    args, _ = ap.parse_known_args()
    
    print("\n" + "=" * 60)
    print("  STAGE: DEM Terrain Extraction")
    print("=" * 60)
    
    config = get_config()
    buffer_m = config.get("terrain", {}).get("buffer_m", 250)
    
    gj_path = DATA_PROCESSED / "india_glacial_lakes_2022.geojson"
    if not gj_path.exists():
        print(f"Error: {gj_path} not found.")
        sys.exit(1)
        
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
        return
        
    cached_tiles = list(DEM_RAW_DIR.glob("*.tif"))
    if not cached_tiles:
        print("Warning: No cached DEM tiles found in data/raw/dem/")
        
    results = []
    for idx, row in gdf.iterrows():
        print(f"  Processing {row['lake_uid']}...")
        res = process_lake(row['lake_uid'], row['geometry'], buffer_m, cached_tiles, args.run_id)
        if res:
            results.append(res)
            
    if len(results) > 0:
        df_out = pd.DataFrame(results)
        out_path = DATA_PROCESSED / "lake_terrain.parquet"
        temp_path = DATA_PROCESSED / "lake_terrain.tmp.parquet"
        
        if out_path.exists():
            existing_df = pd.read_parquet(out_path)
            combined = pd.concat([existing_df, df_out], ignore_index=True)
            df_out = combined.drop_duplicates(subset=["observation_id"], keep="last")
            
        df_out.to_parquet(temp_path, index=False)
        temp_path.replace(out_path)
        print(f"\nSaved {len(df_out)} terrain records to {out_path.name}")
        print("  [STATUS] SUCCESS")
    else:
        print("\nNo terrain records generated.")
        print("  [STATUS] FAILED")
    
if __name__ == "__main__":
    main()
