#!/usr/bin/env python3
import os
import argparse
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from pathlib import Path
import json

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    
    print("\n" + "=" * 60)
    print("  STAGE: Match GLOF Events to Step 1 Lakes")
    print("=" * 60)
    
    events_path = PROCESSED_DIR / "glof_events.parquet"
    lakes_path = PROCESSED_DIR / "india_glacial_lakes_2022.geojson"
    
    if not events_path.exists() or not lakes_path.exists():
        print("  [ERROR] Missing required input files.")
        return
        
    df_events = pd.read_parquet(events_path)
    print(f"  Loaded {len(df_events)} historical events.")
    
    print("  Loading lake inventory...")
    gdf_lakes = gpd.read_file(lakes_path)
    print(f"  Loaded {len(gdf_lakes)} lakes.")
    
    # We need projected CRS for accurate distance calculations
    # Let's use a standard projected CRS for India/Himalaya: EPSG:32644 (UTM 44N) or simply Web Mercator (EPSG:3857)
    # Using 3857 for generic distance buffer approximation
    gdf_lakes_proj = gdf_lakes.to_crs(epsg=3857)
    
    matches = []
    
    stats = {
        "events": len(df_events),
        "polygon_matches": 0,
        "distance_matches": 0,
        "name_matches": 0,
        "unmatched": 0,
        "manual_review": 0,
        "conf_high": 0,
        "conf_medium": 0,
        "conf_low": 0,
        "conf_unverified": 0,
        "match_conf_high": 0,
        "match_conf_medium": 0,
        "match_conf_low": 0
    }
    
    for idx, row in df_events.iterrows():
        lat = row["event_latitude"]
        lon = row["event_longitude"]
        
        match_status = "UNMATCHED"
        match_method = None
        match_distance = None
        lake_uid = None
        lake_match_confidence = None
        candidate_count = 0
        
        # Track confidence stats for event
        evt_conf = row["event_confidence"]
        if evt_conf == "HIGH": stats["conf_high"] += 1
        elif evt_conf == "MEDIUM": stats["conf_medium"] += 1
        elif evt_conf == "LOW": stats["conf_low"] += 1
        else: stats["conf_unverified"] += 1
        
        if pd.notnull(lat) and pd.notnull(lon):
            pt = Point(lon, lat)
            # Create a geodataframe for the point
            pt_gdf = gpd.GeoDataFrame(geometry=[pt], crs="EPSG:4326")
            pt_proj = pt_gdf.to_crs(epsg=3857)
            
            # Tier 1: Polygon intersection
            # Find if point intersects any lake polygon
            intersecting = gpd.sjoin(pt_gdf, gdf_lakes, predicate='intersects')
            if len(intersecting) > 0:
                # Intersects one or more lakes
                candidate_count = len(intersecting)
                if candidate_count == 1:
                    lake_uid = intersecting.iloc[0]["lake_uid"]
                    match_status = "MATCHED"
                    match_method = "polygon_intersection"
                    match_distance = 0.0
                    lake_match_confidence = "HIGH"
                    stats["polygon_matches"] += 1
                    stats["match_conf_high"] += 1
                else:
                    # Multiple intersections (overlapping polygons?), manual review needed
                    match_status = "MANUAL_REVIEW"
                    match_method = "polygon_intersection_ambiguous"
                    stats["manual_review"] += 1
            else:
                # Tier 2: Nearest lake within 5km
                buffer_5km = pt_proj.geometry.buffer(5000)
                buffer_gdf_5km = gpd.GeoDataFrame(geometry=buffer_5km, crs="EPSG:3857")
                
                nearby_5km = gpd.sjoin(gdf_lakes_proj, buffer_gdf_5km, predicate='intersects')
                if len(nearby_5km) > 0:
                    distances = nearby_5km.geometry.distance(pt_proj.geometry.iloc[0]).sort_values()
                    closest_idx = distances.index[0]
                    min_dist = distances.iloc[0]
                    candidate_count = len(nearby_5km)
                    
                    lake_uid = nearby_5km.loc[closest_idx, "lake_uid"]
                    match_method = "distance_buffer_5km"
                    match_distance = round(min_dist, 1)
                    
                    is_dominant = False
                    if candidate_count == 1:
                        is_dominant = True
                    elif candidate_count > 1:
                        second_dist = distances.iloc[1]
                        if second_dist - min_dist > 300:
                            is_dominant = True
                            
                    if is_dominant:
                        match_status = "MATCHED"
                        lake_match_confidence = "MEDIUM"
                        stats["distance_matches"] += 1
                        stats["match_conf_medium"] += 1
                    else:
                        match_status = "MANUAL_REVIEW"
                        lake_match_confidence = "LOW"
                        stats["manual_review"] += 1
                        stats["match_conf_low"] += 1
                else:
                    # Tier 3: Diagnostic 5-10km
                    buffer_10km = pt_proj.geometry.buffer(10000)
                    buffer_gdf_10km = gpd.GeoDataFrame(geometry=buffer_10km, crs="EPSG:3857")
                    nearby_10km = gpd.sjoin(gdf_lakes_proj, buffer_gdf_10km, predicate='intersects')
                    
                    if len(nearby_10km) > 0:
                        match_status = "UNMATCHED"
                        match_method = "diagnostic_missed_match_10km"
                        stats["unmatched"] += 1
                    else:
                        match_status = "UNMATCHED"
                        match_method = "no_candidate_10km"
                        stats["unmatched"] += 1
        else:
            match_status = "UNMATCHED"
            stats["unmatched"] += 1
            
        m = {
            "event_id": row["event_id"],
            "lake_uid": lake_uid,
            "match_status": match_status,
            "match_method": match_method,
            "match_distance_m": match_distance,
            "event_confidence": row["event_confidence"],
            "lake_match_confidence": lake_match_confidence,
            "candidate_count": candidate_count,
            "source": row["source"],
            "notes": "Matched automatically"
        }
        matches.append(m)
        
    df_matches = pd.DataFrame(matches)
    
    # Save parquet
    out_path = PROCESSED_DIR / "glof_event_lake_matches.parquet"
    df_matches.to_parquet(out_path, index=False)
    
    # Generate Report
    report_path = PROCESSED_DIR / "glof_event_matching_report.md"
    with open(report_path, "w") as f:
        f.write("# GlacioGuard Historical GLOF Matching Report\n\n")
        f.write("## Source Coverage\n")
        f.write(f"- Source datasets: 1 (Veh et al. Database V3.1)\n")
        f.write(f"- Time range: Historical\n")
        f.write(f"- Geographic coverage: India/Himalaya\n\n")
        
        f.write("## Event Deduplication\n")
        f.write(f"- Final unique events: {stats['events']}\n\n")
        
        f.write("## Lake Matching\n")
        f.write(f"- Matched events: {stats['polygon_matches'] + stats['distance_matches'] + stats['name_matches']}\n")
        f.write(f"- Unmatched events: {stats['unmatched']}\n")
        f.write(f"- Manual-review events: {stats['manual_review']}\n")
        f.write(f"- Polygon matches: {stats['polygon_matches']}\n")
        f.write(f"- Distance matches: {stats['distance_matches']}\n")
        f.write(f"- Name-assisted matches: {stats['name_matches']}\n\n")
        
        f.write("## Confidence\n")
        f.write("### Event Confidence\n")
        f.write(f"- HIGH: {stats['conf_high']}\n")
        f.write(f"- MEDIUM: {stats['conf_medium']}\n")
        f.write(f"- LOW: {stats['conf_low']}\n")
        f.write(f"- UNVERIFIED: {stats['conf_unverified']}\n\n")
        f.write("### Lake Match Confidence\n")
        f.write(f"- HIGH: {stats['match_conf_high']}\n")
        f.write(f"- MEDIUM: {stats['match_conf_medium']}\n")
        f.write(f"- LOW: {stats['match_conf_low']}\n\n")
        
        f.write("## Uncertainty\n")
        f.write("Many historical GLOF records specify only generic regions or river basins. Precise coordinates are often estimated from post-event satellite imagery. Tier-2 distance matching (up to 5km buffer) flags events for manual review when ambiguous candidate lakes are detected, preventing false-positive assignments. Unmatched events remain in the dataset for potential future reconciliation.\n")
        
    print(f"  Matched {stats['polygon_matches'] + stats['distance_matches']} events to lakes.")
    print(f"  Saved matches to {out_path.name}")
    print(f"  Saved report to {report_path.name}")
    print("  [STATUS] SUCCESS")
    
if __name__ == "__main__":
    main()
