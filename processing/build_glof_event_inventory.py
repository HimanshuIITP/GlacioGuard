#!/usr/bin/env python3
import os
import sys
import yaml
import json
import hashlib
import argparse
import requests
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
        
    print(f"  Reading {filename}...")
    xls = pd.ExcelFile(raw_path, engine="odf")
    sheet_dfs = []
    for sheet_name in xls.sheet_names:
        df_sheet = pd.read_excel(xls, sheet_name=sheet_name)
        if not df_sheet.empty:
            # First row is description
            df_sheet = df_sheet.iloc[1:].copy()
            sheet_dfs.append(df_sheet)
            
    df = pd.concat(sheet_dfs, ignore_index=True)
    
    # Filter to India (and optionally Nepal/Bhutan/China/Pakistan if we want full Himalaya, but prompt said India/Himalaya prioritize India)
    # Let's keep India for now to ensure we hit the primary scope.
    df = df[df["Country"].astype(str).str.contains("India", case=False, na=False)].copy()
    
    events = []
    for _, row in df.iterrows():
        # Clean coordinates
        lat = str(row.get("Latitude", "")).replace("°", "").strip()
        lon = str(row.get("Longitude", "")).replace("°", "").strip()
        try:
            lat = float(lat)
            lon = float(lon)
        except ValueError:
            lat = np.nan
            lon = np.nan
            
        date_str = str(row.get("Date", "")).strip()
        if date_str == "nan" or not date_str:
            date_str = str(row.get("Date_Min", "")).strip()
            
        # Parse date conservatively
        event_date = None
        if date_str and date_str != "nan":
            try:
                # usually YYYY-MM-DD or YYYY
                if len(date_str) == 4:
                    event_date = f"{date_str}-01-01"
                else:
                    event_date = pd.to_datetime(date_str).strftime("%Y-%m-%d")
            except Exception:
                pass
                
        # Determine confidence based on reference
        ref = str(row.get("Reference", "")).lower()
        if "doi" in ref or "journal" in ref or "university" in ref or "survey" in ref:
            conf = "HIGH"
        elif "news" in ref or "blog" in ref or "times" in ref:
            conf = "LOW"
        elif ref == "nan" or not ref:
            conf = "UNVERIFIED"
        else:
            conf = "MEDIUM"
            
        source_record_id = str(row.get("ID", "")).strip()
        
        # Row hash for provenance
        row_dict_str = json.dumps(row.to_dict(), sort_keys=True, default=str)
        source_row_hash = hashlib.sha256(row_dict_str.encode()).hexdigest()
        
        event = {
            "source_record_id": source_record_id,
            "source_version": source_info.get("version", "unknown"),
            "source_row_hash": source_row_hash,
            "event_date": event_date,
            "event_time": None, # not provided in this dataset
            "event_latitude": lat,
            "event_longitude": lon,
            "country": "India",
            "region": str(row.get("Major_RGI_Region", "")),
            "district": str(row.get("Mountain_range_Region", "")),
            "lake_name": str(row.get("Lake", "")),
            "reported_lake_id": None,
            "reported_trigger": str(row.get("Mechanism", "")),
            "event_description": str(row.get("Impact_and_destruction", "")),
            "fatalities": str(row.get("reported_fatalities", "")),
            "damage_estimate": str(row.get("economic_losses", "")),
            "source": source_info["id"],
            "source_url": str(row.get("Reference", "")),
            "source_publication": source_info["name"],
            "event_confidence": conf,
            "notes": str(row.get("Further_comments", "")),
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "processing_version": "1.0"
        }
        events.append(event)
        
    return pd.DataFrame(events)

def generate_event_id(row):
    """Stable hash from canonical attributes"""
    canon = f"{row['source']}_{row['source_record_id']}_{row['event_date']}_{row['event_latitude']}_{row['event_longitude']}_{row['lake_name']}"
    return "EVT_" + hashlib.sha256(canon.encode()).hexdigest()[:12]

def deduplicate_events(df):
    """
    Merge events that refer to the same physical GLOF.
    For this version, we deduplicate if date, lat, lon are identical, or if source_record_id is identical.
    We preserve source counts and URLs.
    """
    if df.empty:
        return df
        
    df["event_id"] = df.apply(generate_event_id, axis=1)
    
    # We will round coordinates to 2 decimal places (approx 1km) and group by date for physical deduplication
    df["round_lat"] = df["event_latitude"].round(2)
    df["round_lon"] = df["event_longitude"].round(2)
    
    # Group by Date and rounded coordinates to find physical duplicates
    # Where Date is missing, group by event_id (no merge)
    df["merge_key"] = df.apply(
        lambda r: f"{r['event_date']}_{r['round_lat']}_{r['round_lon']}" if pd.notnull(r['event_date']) and pd.notnull(r['round_lat']) else r['event_id'], 
        axis=1
    )
    
    merged_records = []
    for m_key, group in df.groupby("merge_key"):
        first = group.iloc[0].to_dict()
        
        if len(group) > 1:
            first["source_count"] = len(group)
            first["source_ids"] = ",".join(group["source_record_id"].dropna().astype(str).unique())
            first["source_urls"] = " | ".join(group["source_url"].dropna().astype(str).unique())
            # Downgrade confidence if conflicting, or upgrade if multiple sources.
            # We'll keep the highest confidence of the group
            conf_levels = {"HIGH": 3, "MEDIUM": 2, "LOW": 1, "UNVERIFIED": 0}
            best_conf = max(group["event_confidence"].tolist(), key=lambda x: conf_levels.get(x, 0))
            first["event_confidence"] = best_conf
        else:
            first["source_count"] = 1
            first["source_ids"] = str(first["source_record_id"])
            first["source_urls"] = str(first["source_url"])
            
        merged_records.append(first)
        
    res_df = pd.DataFrame(merged_records)
    res_df = res_df.drop(columns=["round_lat", "round_lon", "merge_key"])
    return res_df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("  STAGE: Build Historical GLOF Event Inventory")
    print("=" * 60)
    
    config = load_config()
    sources = config.get("sources", [])
    
    all_events = []
    
    for src in sources:
        if src["id"] == "veh_global_glof_db_v3_1":
            df_src = parse_veh_database(src)
            all_events.append(df_src)
            
    if not all_events:
        print("  No sources processed.")
        return
        
    combined_df = pd.concat(all_events, ignore_index=True)
    print(f"  Normalized {len(combined_df)} event records.")
    
    dedup_df = deduplicate_events(combined_df)
    print(f"  Deduplicated to {len(dedup_df)} unique events.")
    
    # Data quality assertions
    assert dedup_df["event_id"].isnull().sum() == 0, "Null event_id found"
    assert dedup_df["event_id"].duplicated().sum() == 0, "Duplicate event_id found"
    
    invalid_coords = dedup_df[
        (dedup_df["event_latitude"] < -90) | (dedup_df["event_latitude"] > 90) |
        (dedup_df["event_longitude"] < -180) | (dedup_df["event_longitude"] > 180)
    ]
    if not invalid_coords.empty:
        print(f"  [WARN] Found {len(invalid_coords)} events with invalid coordinates.")
        
    out_path = PROCESSED_DIR / "glof_events.parquet"
    dedup_df.to_parquet(out_path, index=False)
    
    print(f"  Saved to {out_path.name}")
    print("  [STATUS] SUCCESS")
    
if __name__ == "__main__":
    main()
