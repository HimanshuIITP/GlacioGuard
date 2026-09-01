#!/usr/bin/env python3
"""
GlacioGuard — Shared Spatial Extraction Utilities
Provides dynamic CRS calculation, circular statistics, and reusable raster masking.
"""

import math
import numpy as np
import geopandas as gpd

def get_local_utm_crs(longitude, latitude):
    """
    Returns the appropriate EPSG code for the UTM zone based on longitude and latitude.
    For India/Himalayas, this is typically in the northern hemisphere (EPSG 326XX).
    """
    utm_zone = math.floor((longitude + 180) / 6) + 1
    if latitude >= 0:
        epsg_code = 32600 + utm_zone
    else:
        epsg_code = 32700 + utm_zone
    return f"EPSG:{epsg_code}"

def circular_mean(angles_deg):
    """
    Computes the mean of angles in degrees, properly handling circularity 
    (e.g., mean of 1 and 359 is 0, not 180).
    """
    if len(angles_deg) == 0:
        return np.nan
        
    angles_rad = np.radians(angles_deg)
    mean_sin = np.nanmean(np.sin(angles_rad))
    mean_cos = np.nanmean(np.cos(angles_rad))
    
    mean_angle_rad = np.arctan2(mean_sin, mean_cos)
    mean_angle_deg = np.degrees(mean_angle_rad)
    
    if mean_angle_deg < 0:
        mean_angle_deg += 360
        
    return mean_angle_deg

def get_dem_tile_name(lat, lon):
    """
    Returns the deterministic 1x1 degree SRTMGL1 tile name for a given latitude and longitude.
    """
    s = math.floor(lat)
    w = math.floor(lon)
    return f"SRTMGL1_N{s}_E{w}.tif"
