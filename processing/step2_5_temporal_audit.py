#!/usr/bin/env python3
import pandas as pd
from pathlib import Path
import json

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

def get_observation_coverage(lake_uid, start_dt, end_dt):
    """Calculate valid reference rows for a specific lake and window."""
    w_path = PROCESSED_DIR / "lake_weather_observations.parquet"
    if not w_path.exists():
        return 0
    df_w = pd.read_parquet(w_path)
    sub = df_w[(df_w["lake_uid"] == lake_uid)].copy()
    if sub.empty:
        return 0
    sub["observation_timestamp"] = pd.to_datetime(sub["observation_timestamp"], utc=True)
    mask = (sub["observation_timestamp"] >= start_dt) & (sub["observation_timestamp"] <= end_dt)
    return len(sub[mask])

def main():
    print("Running Step 2.5.5 Temporal Event Eligibility Audit...")
    
    events_path = PROCESSED_DIR / "glof_events.parquet"
    matches_path = PROCESSED_DIR / "glof_event_lake_matches.parquet"
    
    df_events = pd.read_parquet(events_path)
    df_matches = pd.read_parquet(matches_path)
    
    # 1. Spatially confirmed
    df_full = df_events.merge(df_matches, on="event_id", how="inner")
    mask_confirmed = (df_full["match_status"] == "HIGH_CONFIDENCE_MATCH") & (df_full["event_confidence_x"].isin(["HIGH", "MEDIUM"]))
    df_conf = df_full[mask_confirmed].copy()
    
    out_rows = []
    
    # Track stats
    stats = {
        "total_confirmed": len(df_conf),
        "new_confirmed": 0,
        "new_valid_date": 0,
        "new_no_date": 0,
        "new_with_obs": 0,
        "new_unlabelable": 0
    }
    
    # Old ones had valid dates (4 events from the baseline)
    # We can detect them by simply sorting and looking at which had data originally, or checking if their date is perfectly known.
    # We know the old 4 events are GLACIOGUARD_IN_000162, GLACIOGUARD_IN_001426, GLACIOGUARD_IN_001515, GLACIOGUARD_IN_002204
    old_lakes = {"GLACIOGUARD_IN_000162", "GLACIOGUARD_IN_001426", "GLACIOGUARD_IN_001515", "GLACIOGUARD_IN_002204"}
    
    report_rows = []
    
    for idx, row in df_conf.iterrows():
        eid = row["event_id"]
        lake_uid = row["lake_uid"]
        edate = row["event_date"]
        
        is_new = lake_uid not in old_lakes
        if is_new:
            stats["new_confirmed"] += 1
            
        status = "LABELABLE"
        reason = "Valid date and high confidence match"
        
        # 2. Check Date Valid
        date_str = str(edate)
        is_valid_date = False
        event_dt = None
        
        if date_str != "None" and date_str != "nan":
            if len(date_str) == 10:
                event_dt = pd.to_datetime(date_str + " 00:00:00", utc=True, errors="coerce")
            else:
                event_dt = pd.to_datetime(date_str, utc=True, errors="coerce")
                
            if pd.notna(event_dt):
                is_valid_date = True
        
        # Check source fields for potential exact date if missing
        if not is_valid_date:
            try:
                orig = json.loads(row["original_event_fields"])
            except:
                orig = {}
            # If the original field just has '1971' or '2023', we cannot infer exact date safely without fabricating.
            reason = "No exact event date available in source"
            status = "TEMPORALLY_UNLABELABLE"
            
            if is_new:
                stats["new_no_date"] += 1
                stats["new_unlabelable"] += 1
        else:
            if is_new:
                stats["new_valid_date"] += 1
                
        # 3. Check historical coverage if labelable
        valid_rows = 0
        window_start, window_end = None, None
        
        if is_valid_date:
            window_start = event_dt - pd.Timedelta(days=45)
            window_end = event_dt + pd.Timedelta(days=45)
            valid_rows = get_observation_coverage(lake_uid, window_start, window_end)
            
            if valid_rows == 0:
                # Still temporally unlabelable if there's no data supporting the window
                status = "TEMPORALLY_UNLABELABLE"
                reason = "No historical weather observations available for the 90-day window"
                if is_new:
                    stats["new_unlabelable"] += 1
            else:
                if is_new:
                    stats["new_with_obs"] += 1

        out_rows.append({
            "event_id": eid,
            "lake_uid": lake_uid,
            "event_date": edate if is_valid_date else None,
            "event_time": row["event_time"],
            "spatial_match_status": row["match_status"],
            "temporal_eligibility_status": status,
            "reason": reason,
            "processing_version": "1.0"
        })
        
        report_rows.append({
            "lake_uid": lake_uid,
            "status": status,
            "reason": reason,
            "date": edate,
            "coverage": valid_rows
        })
        
    df_out = pd.DataFrame(out_rows)
    out_path = PROCESSED_DIR / "step2_5_temporal_event_eligibility.parquet"
    df_out.to_parquet(out_path, index=False)
    
    # Report
    md_path = PROCESSED_DIR / "step2_5_temporal_event_eligibility_report.md"
    with open(md_path, "w") as f:
        f.write("# Step 2.5.5 Temporal Event Eligibility Audit\n\n")
        f.write("## Overview\n")
        f.write(f"- Spatially confirmed events: {len(df_conf)}\n")
        
        labelable_count = len(df_out[df_out["temporal_eligibility_status"] == "LABELABLE"])
        unlabelable_count = len(df_out[df_out["temporal_eligibility_status"] == "TEMPORALLY_UNLABELABLE"])
        
        f.write(f"- Temporally labelable events (Ready for Step 3.2): {labelable_count}\n")
        f.write(f"- Temporally unlabelable events: {unlabelable_count}\n\n")
        
        f.write("## Event Details\n")
        f.write("| Lake UID | Event Date | Status | Reason | Ref Rows |\n")
        f.write("|----------|------------|--------|--------|----------|\n")
        for r in report_rows:
            f.write(f"| {r['lake_uid']} | {r['date']} | {r['status']} | {r['reason']} | {r['coverage']} |\n")
            
        f.write("\n## New Events Summary (Total 10)\n")
        f.write(f"- New confirmed events with valid dates: {stats['new_valid_date']}\n")
        f.write(f"- New confirmed events without dates: {stats['new_no_date']}\n")
        f.write(f"- New events with sufficient historical obs: {stats['new_with_obs']}\n")
        f.write(f"- New events that remain temporally unlabelable: {stats['new_unlabelable']}\n\n")
        
    print(f"  Saved {out_path.name}")
    print(f"  Saved {md_path.name}")
    
if __name__ == "__main__":
    main()
