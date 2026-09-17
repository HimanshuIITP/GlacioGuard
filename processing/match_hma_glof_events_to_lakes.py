import sys
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from pathlib import Path
import difflib

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

def main():
    print("=" * 60)
    print("  STAGE: Match HMA Events to Lakes")
    print("=" * 60)
    
    # 1. Load data
    events_df = pd.read_parquet(PROCESSED_DIR / "hma_event_inventory.parquet")
    lakes_df = pd.read_parquet(PROCESSED_DIR / "hma_lake_inventory_registry.parquet")
    
    # Ensure geometries
    gdf_lakes = gpd.GeoDataFrame(
        lakes_df, 
        geometry=gpd.GeoSeries.from_wkb(lakes_df['geometry']) if 'geometry' in lakes_df.columns and type(lakes_df['geometry'].iloc[0]) == bytes else lakes_df['geometry'],
        crs="EPSG:4326"
    )
    
    events_gdf = gpd.GeoDataFrame(
        events_df, 
        geometry=gpd.points_from_xy(events_df.longitude, events_df.latitude),
        crs="EPSG:4326"
    )
    
    # We need to project to a metric CRS for accurate distance calculations
    # UTM zone 45N (EPSG:32645) is a reasonable approximation for Nepal/Bhutan.
    events_gdf = events_gdf.to_crs(epsg=32645)
    gdf_lakes = gdf_lakes.to_crs(epsg=32645)
    
    matches = []
    
    for idx, event in events_gdf.iterrows():
        event_geom = event.geometry
        event_country = event['country']
        
        # Strictly pair by country
        country_lakes = gdf_lakes[gdf_lakes['country'] == event_country].copy()
        
        if country_lakes.empty:
            matches.append({
                'event_id': event['event_id'],
                'lake_uid': None,
                'match_status': 'UNMATCHED',
                'nearest_lake_distance': None,
                'lake_name_evidence': False,
                'historical_source_lake_identifier': event.get('reported_lake_id'),
                'inventory_epoch': None,
                'spatially_plausible_no_intersection': False
            })
            continue
            
        # Calculate distances to all lakes in that country
        country_lakes['distance_to_event'] = country_lakes.geometry.distance(event_geom)
        country_lakes = country_lakes.sort_values('distance_to_event')
        
        best_candidate = country_lakes.iloc[0]
        min_dist = best_candidate['distance_to_event']
        
        # Check intersection
        intersecting_lakes = country_lakes[country_lakes['distance_to_event'] == 0]
        
        if not intersecting_lakes.empty:
            best_intersect = intersecting_lakes.iloc[0]
            matches.append({
                'event_id': event['event_id'],
                'lake_uid': best_intersect['lake_uid'],
                'match_status': 'HIGH_CONFIDENCE_MATCH',
                'nearest_lake_distance': 0.0,
                'lake_name_evidence': True,
                'historical_source_lake_identifier': event.get('reported_lake_id'),
                'inventory_epoch': best_intersect['inventory_epoch'],
                'spatially_plausible_no_intersection': False
            })
            continue
            
        # No intersection (Historical Disappearance Logic)
        name_evidence = False
        if pd.notna(event['lake_name']) and pd.notna(best_candidate.get('source_lake_id')):
            sim = difflib.SequenceMatcher(None, str(event['lake_name']).lower(), str(best_candidate.get('source_lake_id')).lower()).ratio()
            if sim > 0.8:
                name_evidence = True
        
        if pd.notna(event.get('reported_lake_id')) and pd.notna(best_candidate.get('source_lake_id')):
            if str(event.get('reported_lake_id')) == str(best_candidate.get('source_lake_id')):
                name_evidence = True
        
        # We also might not have lake names in the ICIMOD inventory, it often just has IDs.
        # If there's no intersection, but it's within 5km AND there's name/ID evidence -> MANUAL_REVIEW
        # OR if we simply assume if it's within 5km it's spatially plausible but we need name evidence to justify review.
        # Actually, let's treat any candidate within 5km as spatially plausible. If it has name evidence -> MANUAL_REVIEW.
        # Otherwise if it's within 5km without name evidence -> UNMATCHED (but preserving distance diagnostic)
        
        spatially_plausible = min_dist <= 5000
        
        if min_dist <= 5000:
            if name_evidence:
                status = 'MANUAL_REVIEW'
            else:
                # If we don't have name evidence, it's just unmatched with diagnostic data
                status = 'UNMATCHED'
        elif min_dist <= 10000:
            status = 'UNMATCHED'
        else:
            status = 'UNMATCHED'
            best_candidate = None
            min_dist = None
            
        matches.append({
            'event_id': event['event_id'],
            'lake_uid': best_candidate['lake_uid'] if best_candidate is not None else None,
            'match_status': status,
            'nearest_lake_distance': min_dist,
            'lake_name_evidence': name_evidence,
            'historical_source_lake_identifier': event.get('reported_lake_id'),
            'inventory_epoch': best_candidate['inventory_epoch'] if best_candidate is not None else None,
            'spatially_plausible_no_intersection': spatially_plausible
        })
        
    df_matches = pd.DataFrame(matches)
    
    out_path = PROCESSED_DIR / "hma_event_lake_matches.parquet"
    df_matches.to_parquet(out_path, index=False)
    
    print(df_matches['match_status'].value_counts())
    print(f"Saved {len(df_matches)} matches to {out_path.name}")
    
    
if __name__ == "__main__":
    main()
