#!/usr/bin/env python3
import os
import sys
import yaml
import json
import hashlib
import argparse
import requests
import zipfile
import shutil
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "config" / "glof_event_sources.yaml"
RAW_DIR = BASE_DIR / "data" / "raw" / "glof_events"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

def load_config():
    if not CONFIG_PATH.exists():
        return {}
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

def download_file(url, out_path):
    print(f"  Downloading {url}...")
    r = requests.get(url)
    r.raise_for_status()
    with open(out_path, "wb") as f:
        f.write(r.content)

def parse_veh_database(source_info):
    url = source_info["url"]
    filename = url.split("/")[-2]
    raw_path = RAW_DIR / filename
    
    if not raw_path.exists():
        download_file(url, raw_path)
        
    print(f"  Reading Veh {filename}...")
    xls = pd.ExcelFile(raw_path, engine="odf")
    sheet_dfs = []
    for sheet_name in xls.sheet_names:
        df_sheet = pd.read_excel(xls, sheet_name=sheet_name)
        if not df_sheet.empty:
            df_sheet = df_sheet.iloc[1:].copy()
            sheet_dfs.append(df_sheet)
            
    df = pd.concat(sheet_dfs, ignore_index=True)
    
    if "Country" in df.columns:
        # Implicit India records
        india_keywords = ["himachal", "uttarakhand", "sikkim", "arunachal", "jammu", "kashmir", "ladakh", "india"]
        def is_india_region(r):
            text = str(r.get("Major_RGI_Region", "")) + " " + str(r.get("Mountain_range_Region", "")) + " " + str(r.get("Lake", "")) + " " + str(r.get("Location", ""))
            return any(kw in text.lower() for kw in india_keywords)

        mask_implicit = ~df["Country"].astype(str).str.contains("India", case=False, na=False) & df.apply(is_india_region, axis=1)
        if mask_implicit.any():
            df.loc[mask_implicit, "Country"] = "India"
                
        df = df[df["Country"].astype(str).str.contains("India", case=False, na=False)].copy()
        
    events = []
    for _, row in df.iterrows():
        lat = str(row.get("Latitude", "")).replace("°", "").strip()
        lon = str(row.get("Longitude", "")).replace("°", "").strip()
        try:
            lat, lon = float(lat), float(lon)
        except ValueError:
            lat, lon = np.nan, np.nan
            
        date_str = str(row.get("Date", "")).strip()
        if date_str == "nan" or not date_str:
            date_str = str(row.get("Date_Min", "")).strip()
            
        event_date = None
        if date_str and date_str != "nan":
            try:
                if len(date_str) == 4:
                    event_date = f"{date_str}-01-01"
                else:
                    event_date = pd.to_datetime(date_str).strftime("%Y-%m-%d")
            except Exception: pass
                
        ref = str(row.get("Reference", "")).lower()
        if "doi" in ref or "journal" in ref or "university" in ref or "survey" in ref:
            conf = "HIGH"
        elif "news" in ref or "blog" in ref or "times" in ref:
            conf = "LOW"
        elif ref == "nan" or not ref:
            conf = "UNVERIFIED"
        else:
            conf = "MEDIUM"
            
        event = {
            "source_record_id": str(row.get("ID", "")).strip(),
            "source_version": source_info.get("version", "unknown"),
            "event_date": event_date,
            "event_time": None,
            "event_latitude": lat,
            "event_longitude": lon,
            "country": "India",
            "region": str(row.get("Major_RGI_Region", "")),
            "district": str(row.get("Mountain_range_Region", "")),
            "lake_name": str(row.get("Lake", "")),
            "reported_trigger": str(row.get("Mechanism", "")),
            "event_description": str(row.get("Impact_and_destruction", "")),
            "fatalities": str(row.get("reported_fatalities", "")),
            "source": source_info["id"],
            "source_url": str(row.get("Reference", "")),
            "event_confidence": conf,
            "original_event_fields": json.dumps(row.to_dict(), default=str)
        }
        events.append(event)
        
    return pd.DataFrame(events)

def parse_icimod_database(source_info):
    url = source_info["url"]
    icimod_dir = RAW_DIR / "icimod"
    zip_path = icimod_dir / "icimod.zip"
    
    if not icimod_dir.exists():
        icimod_dir.mkdir(parents=True)
        download_file(url, zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(icimod_dir)
            
    # Schema Discovery
    candidate_csvs = list(icimod_dir.rglob("*.csv"))
    if not candidate_csvs:
        print("  [ERROR] No CSV found in ICIMOD archive.")
        return pd.DataFrame()
        
    # Find the authoritative table
    best_df = None
    best_score = 0
    for csv_file in candidate_csvs:
        try:
            df = pd.read_csv(csv_file, encoding='ISO-8859-1', low_memory=False)
            cols = [c.lower() for c in df.columns]
            score = sum(1 for kw in ['lat', 'lon', 'lake', 'year', 'date', 'country'] if any(kw in c for c in cols))
            if score > best_score:
                best_score = score
                best_df = df
        except Exception: pass
        
    if best_df is None:
        print("  [ERROR] Could not parse ICIMOD CSVs.")
        return pd.DataFrame()
        
    df = best_df
    print(f"  Selected ICIMOD table with {len(df)} rows.")
    
    # India Filter
    india_keywords = ["himachal", "uttarakhand", "sikkim", "arunachal", "jammu", "kashmir", "ladakh", "india"]
    
    def is_india(row):
        c = str(row.get("Country", "")).lower()
        if "india" in c: return True
        r1 = str(row.get("Province", "")).lower()
        r2 = str(row.get("Region_RGI", "")).lower()
        if any(kw in r1 for kw in india_keywords) or any(kw in r2 for kw in india_keywords):
            return True
        return False
        
    df_india = df[df.apply(is_india, axis=1)].copy()
    print(f"  ICIMOD India records: {len(df_india)}")
    
    events = []
    for _, row in df_india.iterrows():
        lat = row.get("Lat_lake", np.nan)
        lon = row.get("Lon_lake", np.nan)
        
        try: lat, lon = float(lat), float(lon)
        except: lat, lon = np.nan, np.nan
        
        y = row.get("Year_exact", row.get("Year_approx"))
        m = row.get("Month", np.nan)
        d = row.get("Day", np.nan)
        
        event_date = None
        try:
            if pd.notnull(y):
                y = int(float(y))
                if pd.notnull(m) and pd.notnull(d):
                    event_date = f"{y}-{int(float(m)):02d}-{int(float(d)):02d}"
                else:
                    event_date = f"{y}-01-01"
        except: pass
        
        conf = "MEDIUM" # ICIMOD is generally well-curated, but we can look for specific flags
        if pd.notnull(row.get("Ref_scientific")):
            conf = "HIGH"
            
        event = {
            "source_record_id": str(row.get("GF_ID", "")),
            "source_version": source_info.get("version", "unknown"),
            "event_date": event_date,
            "event_time": None,
            "event_latitude": lat,
            "event_longitude": lon,
            "country": "India",
            "region": str(row.get("Province", "")),
            "district": str(row.get("River_Basin", "")),
            "lake_name": str(row.get("Lake_name", "")),
            "reported_trigger": str(row.get("Driver_lake", "")) + " / " + str(row.get("Mechanism", "")),
            "event_description": str(row.get("Impact", "")),
            "fatalities": str(row.get("Lives_total", "")),
            "source": source_info["id"],
            "source_url": str(row.get("Ref_scientific_full", "")),
            "event_confidence": conf,
            "original_event_fields": json.dumps(row.to_dict(), default=str)
        }
        events.append(event)
        
    return pd.DataFrame(events)

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0 # km
    dlat, dlon = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

def generate_event_id(row):
    canon = f"{row['source']}_{row['source_record_id']}_{row['event_date']}_{row['event_latitude']}_{row['event_longitude']}_{row['lake_name']}"
    return "EVT_" + hashlib.sha256(canon.encode()).hexdigest()[:12]

def cross_source_dedup(df):
    if df.empty: return df, pd.DataFrame()
    
    df["event_id"] = df.apply(generate_event_id, axis=1)
    
    processed_indices = set()
    canonical_events = []
    reconciliation = []
    
    for i, row1 in df.iterrows():
        if i in processed_indices: continue
        
        duplicate_group = [row1.to_dict()]
        processed_indices.add(i)
        dedup_status = "NEW_EVENT"
        
        # Look for matches in the rest of the dataframe
        for j, row2 in df.iterrows():
            if j in processed_indices: continue
            
            # Deterministic Scoring
            score = 0
            
            # 1. Coordinate Distance
            dist = 9999
            if pd.notnull(row1["event_latitude"]) and pd.notnull(row2["event_latitude"]):
                dist = haversine(row1["event_latitude"], row1["event_longitude"], row2["event_latitude"], row2["event_longitude"])
                if dist < 2.0: score += 3
                elif dist < 10.0: score += 1
                
            # 2. Date Proximity
            date_match = False
            if pd.notnull(row1["event_date"]) and pd.notnull(row2["event_date"]):
                y1 = str(row1["event_date"])[:4]
                y2 = str(row2["event_date"])[:4]
                if str(row1["event_date"]) == str(row2["event_date"]):
                    score += 4
                    date_match = True
                elif y1 == y2:
                    score += 2
                    date_match = True
                    
            # 3. Lake Name / Identity
            n1 = str(row1["lake_name"]).lower().strip()
            n2 = str(row2["lake_name"]).lower().strip()
            if n1 and n2 and n1 != "nan" and n2 != "nan" and n1 != "unknown" and n1 == n2:
                score += 3
                
            if score >= 6 or (dist < 2.0 and date_match):
                # Strong Match -> Merge
                duplicate_group.append(row2.to_dict())
                processed_indices.add(j)
                dedup_status = "DUPLICATE_EXISTING_EVENT"
            elif score >= 3:
                # Ambiguous -> Flag for review but do NOT merge automatically
                reconciliation.append({
                    "canonical_event_id": row1["event_id"],
                    "source_name": row2["source"],
                    "source_record_id": row2["source_record_id"],
                    "duplicate_group_id": row1["event_id"],
                    "dedup_status": "CROSS_SOURCE_REVIEW",
                    "event_date": row2["event_date"],
                    "event_latitude": row2["event_latitude"],
                    "event_longitude": row2["event_longitude"],
                    "lake_name": row2["lake_name"]
                })
        
        # Merge the group
        first = duplicate_group[0]
        if len(duplicate_group) > 1:
            first["source_count"] = len(duplicate_group)
            first["source_ids"] = ",".join([str(x["source_record_id"]) for x in duplicate_group])
            first["source_urls"] = " | ".join([str(x["source_url"]) for x in duplicate_group])
            # Keep highest confidence
            conf_levels = {"HIGH": 3, "MEDIUM": 2, "LOW": 1, "UNVERIFIED": 0}
            best_conf = max([x["event_confidence"] for x in duplicate_group], key=lambda x: conf_levels.get(x, 0))
            first["event_confidence"] = best_conf
            
            # Log all duplicates in reconciliation
            for d in duplicate_group:
                reconciliation.append({
                    "canonical_event_id": first["event_id"],
                    "source_name": d["source"],
                    "source_record_id": d["source_record_id"],
                    "duplicate_group_id": first["event_id"],
                    "dedup_status": dedup_status,
                    "event_date": d["event_date"],
                    "event_latitude": d["event_latitude"],
                    "event_longitude": d["event_longitude"],
                    "lake_name": d["lake_name"]
                })
        else:
            first["source_count"] = 1
            first["source_ids"] = str(first["source_record_id"])
            first["source_urls"] = str(first["source_url"])
            reconciliation.append({
                "canonical_event_id": first["event_id"],
                "source_name": first["source"],
                "source_record_id": first["source_record_id"],
                "duplicate_group_id": first["event_id"],
                "dedup_status": "NEW_EVENT",
                "event_date": first["event_date"],
                "event_latitude": first["event_latitude"],
                "event_longitude": first["event_longitude"],
                "lake_name": first["lake_name"]
            })
            
        canonical_events.append(first)
        
    return pd.DataFrame(canonical_events), pd.DataFrame(reconciliation)

def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("  STAGE: Build Historical GLOF Event Inventory (Multi-Source)")
    print("=" * 60)
    
    config = load_config()
    sources = config.get("sources", [])
    
    all_events = []
    for src in sources:
        if src["id"] == "veh_global_glof_db_v3_1":
            all_events.append(parse_veh_database(src))
        elif src["id"] == "icimod_hma_glof_db":
            all_events.append(parse_icimod_database(src))
            
    if not all_events:
        print("  No sources processed.")
        return
        
    combined_df = pd.concat(all_events, ignore_index=True)
    print(f"  Total multi-source normalized records: {len(combined_df)}")
    
    dedup_df, recon_df = cross_source_dedup(combined_df)
    print(f"  Deduplicated to {len(dedup_df)} unique canonical events.")
    
    # Save outputs
    out_path = PROCESSED_DIR / "glof_events.parquet"
    dedup_df.to_parquet(out_path, index=False)
    
    recon_path = PROCESSED_DIR / "step5_5_multisource_event_reconciliation.parquet"
    recon_df.to_parquet(recon_path, index=False)
    
    print(f"  Saved events to {out_path.name}")
    print(f"  Saved reconciliation to {recon_path.name}")
    print("  [STATUS] SUCCESS")
    
if __name__ == "__main__":
    main()
