import sys
import yaml
import geopandas as gpd
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw" / "glof_events" / "hma"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
CONFIG_PATH = BASE_DIR / "config" / "hma_lake_inventory_sources.yaml"

def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

def standardize_gdf(gdf, country, prefix, id_col, conf):
    # Ensure geometry is valid
    gdf['geometry'] = gdf['geometry'].make_valid()
    
    # Generate stable lake_uids
    # Using row index as fallback if id_col doesn't exist but we know it does
    if id_col in gdf.columns:
        source_ids = gdf[id_col].astype(str).str.zfill(6)
    else:
        source_ids = pd.Series([str(i).zfill(6) for i in range(len(gdf))], index=gdf.index)
        
    gdf['lake_uid'] = f"GLACIOGUARD_{prefix}_" + source_ids
    gdf['country'] = country
    gdf['source_lake_id'] = gdf[id_col].astype(str) if id_col in gdf.columns else "UNKNOWN"
    gdf['inventory_version'] = conf['inventory_version']
    gdf['inventory_epoch'] = conf['inventory_epoch']
    gdf['observation_period'] = conf['observation_period']
    gdf['source_method'] = conf['source_method']
    gdf['source_url'] = conf['source_url']
    gdf['processing_version'] = "1.0"
    
    return gdf

def main():
    print("=" * 60)
    print("  STAGE: Build HMA Lake Inventories")
    print("=" * 60)
    
    config = load_config()['lake_inventories']
    
    # 1. Process Nepal
    nepal_path = RAW_DIR / "icimod_nepal_glacial_lake_2015.geojson"
    print(f"Loading Nepal from {nepal_path.name}...")
    gdf_nepal = gpd.read_file(nepal_path)
    gdf_nepal = standardize_gdf(gdf_nepal, "Nepal", "NP", "GL_ID", config['nepal'])
    nepal_out = PROCESSED_DIR / "nepal_glacial_lakes.geojson"
    gdf_nepal.to_file(nepal_out, driver="GeoJSON")
    print(f"Saved {len(gdf_nepal)} Nepal lakes to {nepal_out.name}")
    
    # 2. Process Bhutan
    bhutan_path = RAW_DIR / "icimod_bhutan_glacial_lake_2005.geojson"
    print(f"Loading Bhutan from {bhutan_path.name}...")
    gdf_bhutan = gpd.read_file(bhutan_path)
    gdf_bhutan = standardize_gdf(gdf_bhutan, "Bhutan", "BT", "ID", config['bhutan'])
    bhutan_out = PROCESSED_DIR / "bhutan_glacial_lakes.geojson"
    gdf_bhutan.to_file(bhutan_out, driver="GeoJSON")
    print(f"Saved {len(gdf_bhutan)} Bhutan lakes to {bhutan_out.name}")
    
    # 3. Process India
    india_path = PROCESSED_DIR / "india_glacial_lakes_2022.geojson"
    print(f"Loading India from {india_path.name}...")
    gdf_india = gpd.read_file(india_path)
    
    # Explicit rule: Do not rewrite Indian lake_uids. Preserve intact.
    # We only add the unified registry metadata columns.
    gdf_india['country'] = "India"
    if 'source_lake_id' not in gdf_india.columns:
        gdf_india['source_lake_id'] = gdf_india['lake_uid']
        
    gdf_india['inventory_version'] = config['india']['inventory_version']
    gdf_india['inventory_epoch'] = config['india']['inventory_epoch']
    gdf_india['observation_period'] = config['india']['observation_period']
    gdf_india['source_method'] = config['india']['source_method']
    gdf_india['source_url'] = config['india']['source_url']
    if 'processing_version' not in gdf_india.columns:
        gdf_india['processing_version'] = "1.0"
    
    print(f"Loaded {len(gdf_india)} India lakes. Intact lake_uid preserved.")
    
    # 4. Generate Unified HMA Lake Registry
    # Keep only the standardized schema columns
    schema_cols = [
        'lake_uid', 'country', 'source_lake_id', 'inventory_version', 
        'inventory_epoch', 'observation_period', 'source_method', 
        'geometry', 'source_url', 'processing_version'
    ]
    
    df_nepal_reg = gdf_nepal[schema_cols]
    df_bhutan_reg = gdf_bhutan[schema_cols]
    df_india_reg = gdf_india[schema_cols]
    
    gdf_registry = pd.concat([df_nepal_reg, df_bhutan_reg, df_india_reg], ignore_index=True)
    registry_out = PROCESSED_DIR / "hma_lake_inventory_registry.parquet"
    
    # Convert to DataFrame (dropping geometry or wkb encoding it for parquet)
    # GeoPandas has built-in to_parquet
    gpd.GeoDataFrame(gdf_registry, geometry='geometry').to_parquet(registry_out, index=False)
    
    print(f"Unified registry built with {len(gdf_registry)} total lakes.")
    print(f"Saved to {registry_out.name}")
    print("\nSummary:")
    print(gdf_registry['country'].value_counts())

if __name__ == "__main__":
    main()
