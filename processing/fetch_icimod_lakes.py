import sys
import json
import urllib.request
import urllib.parse
import ssl
from datetime import datetime, timezone
from pathlib import Path
import geopandas as gpd
import pandas as pd
from shapely.geometry import shape, mapping

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw" / "glof_events" / "hma"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

def fetch_arcgis_layer_geojson(base_url, layer_name):
    print(f"Fetching {layer_name} from {base_url}...")
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    
    # Get all Object IDs
    params = {
        "where": "1=1",
        "returnIdsOnly": "true",
        "f": "json"
    }
    query_url = f"{base_url}/query?" + urllib.parse.urlencode(params)
    try:
        req = urllib.request.Request(query_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, context=ctx) as response:
            data = json.loads(response.read().decode('utf-8'))
    except Exception as e:
        raise RuntimeError(f"Failed to fetch IDs for {layer_name}: {e}")
        
    object_ids = data.get("objectIds", [])
    if not object_ids:
        print(f"No object IDs returned for {layer_name}.")
        object_ids = []
        
    print(f"Total features to fetch: {len(object_ids)}")
    
    chunk_size = 500
    all_features = []
    
    for i in range(0, len(object_ids), chunk_size):
        chunk = object_ids[i:i+chunk_size]
        oid_str = ",".join(map(str, chunk))
        
        params = {
            "objectIds": oid_str,
            "outFields": "*",
            "returnGeometry": "true",
            "f": "geojson",
            "outSR": "4326"
        }
        
        query_url = f"{base_url}/query"
        post_data = urllib.parse.urlencode(params).encode('utf-8')
        req = urllib.request.Request(query_url, data=post_data, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, context=ctx) as response:
            chunk_data = json.loads(response.read().decode('utf-8'))
            
        features = chunk_data.get("features", [])
        all_features.extend(features)
        print(f"  Fetched {len(all_features)} / {len(object_ids)} features...")
        
    # Reconstruct a valid FeatureCollection
    fc = {
        "type": "FeatureCollection",
        "name": layer_name,
        "crs": { "type": "name", "properties": { "name": "urn:ogc:def:crs:OGC:1.3:CRS84" } },
        "features": all_features,
        "metadata": {
            "source_url": base_url,
            "retrieval_timestamp": datetime.now(timezone.utc).isoformat()
        }
    }
    return fc

def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    
    report_lines = [
        "# Step 5.6 Phase 1: Lake Source Acquisition Report\n"
    ]
    
    # 1. Fetch Nepal 2015
    nepal_url = "https://geoapps.icimod.org/icimodarcgis/rest/services/HKH/GlacialLake/MapServer/1"
    try:
        nepal_fc = fetch_arcgis_layer_geojson(nepal_url, "Glacial_Lake_2015_Nepal")
        nepal_path = RAW_DIR / "icimod_nepal_glacial_lake_2015.geojson"
        with open(nepal_path, "w") as f:
            json.dump(nepal_fc, f)
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
        
    # 2. Fetch HKH 2005
    hkh_url = "https://geoapps.icimod.org/icimodarcgis/rest/services/HKH/GlacialLake/MapServer/0"
    try:
        hkh_fc = fetch_arcgis_layer_geojson(hkh_url, "Glacial_Lake_2005_HKH")
        hkh_path = RAW_DIR / "icimod_hkh_glacial_lake_2005.geojson"
        with open(hkh_path, "w") as f:
            json.dump(hkh_fc, f)
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
        
    # 3. Fetch Bhutan Boundary
    bhutan_url = "https://geoapps.icimod.org/icimodarcgis/rest/services/Bhutan/Basemap/MapServer/0"
    try:
        bhutan_bound_fc = fetch_arcgis_layer_geojson(bhutan_url, "Bhutan_Boundary")
        bhutan_bound_path = RAW_DIR / "icimod_bhutan_boundary.geojson"
        with open(bhutan_bound_path, "w") as f:
            json.dump(bhutan_bound_fc, f)
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
        
    # 4. Load into GeoPandas for validation & clipping
    gdf_nepal = gpd.read_file(nepal_path)
    gdf_hkh = gpd.read_file(hkh_path)
    gdf_bhutan_bound = gpd.read_file(bhutan_bound_path)
    
    # For Nepal, load Natural Earth to verify intersection
    ne_path = BASE_DIR / "data" / "raw" / "natural_earth" / "ne_10m_admin_0_countries.shp"
    if ne_path.exists():
        gdf_ne = gpd.read_file(ne_path)
        gdf_ne_nepal = gdf_ne[gdf_ne["ADMIN"] == "Nepal"]
        if not gdf_ne_nepal.empty:
            nepal_geom = gdf_ne_nepal.geometry.iloc[0]
            # Verify Nepal polygons intersect the Nepal geographic boundary
            # Due to precision issues, we might just check distance or slight buffer
            mask = gdf_nepal.intersects(nepal_geom.buffer(0.1))
            invalid_nepal = len(gdf_nepal) - mask.sum()
            if invalid_nepal > 0:
                print(f"[WARNING] {invalid_nepal} Nepal polygons did not strongly intersect the NE Nepal boundary.")
            gdf_nepal = gdf_nepal[mask].copy()
            
    # For Bhutan, clip HKH against the downloaded Bhutan boundary
    if gdf_bhutan_bound.empty:
        print("[ERROR] Bhutan boundary has no geometry.")
        sys.exit(1)
        
    bhutan_geom = gdf_bhutan_bound.geometry.unary_union
    mask_bhutan = gdf_hkh.intersects(bhutan_geom)
    gdf_bhutan = gdf_hkh[mask_bhutan].copy()
    
    # Validation flags
    nepal_valid_geoms = gdf_nepal.is_valid.all()
    bhutan_valid_geoms = gdf_bhutan.is_valid.all()
    
    # Source ID deduplication logic
    # Find the appropriate source ID column
    def validate_ids(gdf, layer_name):
        possible_id_cols = ['GL_ID', 'ID', 'Lake_ID', 'OBJECTID']
        id_col = None
        for col in possible_id_cols:
            if col in gdf.columns:
                id_col = col
                break
                
        if not id_col:
            return "No clear ID column found", 0, 0
            
        null_count = gdf[id_col].isnull().sum()
        dup_count = gdf.duplicated(subset=[id_col]).sum()
        return id_col, null_count, dup_count
        
    nepal_id_col, nepal_nulls, nepal_dups = validate_ids(gdf_nepal, "Nepal")
    bhutan_id_col, bhutan_nulls, bhutan_dups = validate_ids(gdf_bhutan, "Bhutan")
    
    # Save Bhutan file preserving metadata
    bhutan_out = {
        "type": "FeatureCollection",
        "name": "Glacial_Lake_2005_Bhutan_Clipped",
        "crs": { "type": "name", "properties": { "name": "urn:ogc:def:crs:OGC:1.3:CRS84" } },
        "features": json.loads(gdf_bhutan.to_json()).get("features", []),
        "metadata": {
            "source_url": hkh_url,
            "source_layer": "Glacial_Lake_2005_HKH",
            "inventory_epoch": "2005",
            "processing_version": "1.0",
            "clip_source": bhutan_url
        }
    }
    
    bhutan_path = RAW_DIR / "icimod_bhutan_glacial_lake_2005.geojson"
    with open(bhutan_path, "w") as f:
        json.dump(bhutan_out, f)
        
    # Generate Report
    report_lines.append("## Retrieval Details\n")
    report_lines.append(f"**Nepal 2015 Layer:** {nepal_url}")
    report_lines.append(f"- Inventory Epoch: 2015")
    report_lines.append(f"- Raw features fetched: {len(nepal_fc['features'])}")
    report_lines.append(f"- Valid Nepal features (post-boundary verify): {len(gdf_nepal)}")
    report_lines.append(f"- CRS: {gdf_nepal.crs}")
    report_lines.append(f"- All geometries valid: {nepal_valid_geoms}")
    report_lines.append(f"- Source ID column: {nepal_id_col} (Nulls: {nepal_nulls}, Duplicates: {nepal_dups})\n")
    
    report_lines.append(f"**HKH 2005 Layer:** {hkh_url}")
    report_lines.append(f"- Inventory Epoch: 2005")
    report_lines.append(f"- Raw HKH features fetched: {len(hkh_fc['features'])}\n")
    
    report_lines.append(f"**Bhutan Boundary Layer:** {bhutan_url}")
    report_lines.append(f"- Raw boundary features fetched: {len(bhutan_bound_fc['features'])}\n")
    
    report_lines.append("**Bhutan Culling (Intersect HKH 2005 against Bhutan Boundary)**")
    report_lines.append(f"- Bhutan features extracted: {len(gdf_bhutan)}")
    report_lines.append(f"- CRS: {gdf_bhutan.crs}")
    report_lines.append(f"- All geometries valid: {bhutan_valid_geoms}")
    report_lines.append(f"- Source ID column: {bhutan_id_col} (Nulls: {bhutan_nulls}, Duplicates: {bhutan_dups})\n")
    
    report_lines.append("## Download Status: SUCCESS")
    
    report_path = PROCESSED_DIR / "hma_lake_source_acquisition_report.md"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
        
    print(f"\nSaved {report_path.name}")
    print("Lake Source Acquisition Complete.")

if __name__ == "__main__":
    main()
