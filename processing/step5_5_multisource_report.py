#!/usr/bin/env python3
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

def main():
    print("Generating Step 5.5 Multi-Source Report...")
    
    # 1. Load Data
    df_matches = pd.read_parquet(PROCESSED_DIR / "glof_event_lake_matches.parquet")
    df_recon = pd.read_parquet(PROCESSED_DIR / "step5_5_multisource_event_reconciliation.parquet")
    
    diag_path = PROCESSED_DIR / "step5_5_distance_diagnostic_candidates.parquet"
    man_path = PROCESSED_DIR / "step5_5_manual_review_queue.parquet"
    
    df_diag = pd.read_parquet(diag_path) if diag_path.exists() else pd.DataFrame()
    df_man = pd.read_parquet(man_path) if man_path.exists() else pd.DataFrame()
    
    # 2. Source Coverage & Deduplication
    veh_records = len(df_recon[df_recon["source_name"] == "veh_global_glof_db_v3_1"])
    icimod_records = len(df_recon[df_recon["source_name"] == "icimod_hma_glof_db"])
    total_records = veh_records + icimod_records
    
    unique_canonical_events = len(df_matches)
    duplicates = total_records - unique_canonical_events
    cross_source_review = len(df_recon[df_recon["dedup_status"] == "CROSS_SOURCE_REVIEW"])
    new_from_icimod = len(df_recon[(df_recon["source_name"] == "icimod_hma_glof_db") & (df_recon["dedup_status"] == "NEW_EVENT")])
    duplicate_icimod = len(df_recon[(df_recon["source_name"] == "icimod_hma_glof_db") & (df_recon["dedup_status"] == "DUPLICATE_EXISTING_EVENT")])
    
    # 3. Matching Status
    evt_high_conf = len(df_matches[df_matches["match_status"] == "HIGH_CONFIDENCE_MATCH"])
    evt_manual = len(df_matches[df_matches["match_status"] == "MANUAL_REVIEW"])
    evt_unmatched = len(df_matches[df_matches["match_status"] == "UNMATCHED"])
    
    total_diag_candidates = df_diag["diagnostic_candidate_count"].sum() if not df_diag.empty else 0
    total_man_candidates = len(df_man)
    
    # 4. Expansion Comparison
    previous_confirmed = 4 
    new_confirmed = max(0, evt_high_conf - previous_confirmed)
    new_lake_uids = len(df_matches[df_matches["match_status"] == "HIGH_CONFIDENCE_MATCH"]["lake_uid"].unique()) - 4
    
    if new_confirmed > 0:
        status_msg = f"NEW_CONFIRMED_EVENTS = {new_confirmed}"
    else:
        status_msg = "NO_NEW_EVENTS_FROM_SECONDARY_SOURCE"
        
    # 5. Output Report
    report_path = PROCESSED_DIR / "step5_5_multisource_report.md"
    
    with open(report_path, "w") as f:
        f.write("# Step 5.5 Multi-Source Historical GLOF Event Expansion\n\n")
        f.write(f"## OVERALL EXPANSION STATUS\n**{status_msg}**\n\n")
        
        f.write("## 1. Source Coverage (India-Filtered)\n")
        f.write(f"- Veh/Lützow source records: **{veh_records}**\n")
        f.write(f"- ICIMOD source records: **{icimod_records}**\n")
        f.write(f"- Total India-filtered records: **{total_records}**\n\n")
        
        f.write("## 2. Deduplication\n")
        f.write(f"- Total source records parsed: **{total_records}**\n")
        f.write(f"- Duplicate records merged: **{duplicates}**\n")
        f.write(f"- Unique canonical events established: **{unique_canonical_events}**\n")
        f.write(f"- Cross-source review cases: **{cross_source_review}**\n\n")
        
        f.write("## 3. Matching Outcomes\n")
        f.write(f"- HIGH_CONFIDENCE_MATCH (Tier 1): **{evt_high_conf}**\n")
        f.write(f"- MANUAL_REVIEW (Tier 2): **{evt_manual}**\n")
        f.write(f"- UNMATCHED: **{evt_unmatched}**\n")
        f.write(f"- 0–5 km candidate lake rows generated: **{total_man_candidates}**\n")
        f.write(f"- 5–10 km diagnostic candidate rows generated: **{total_diag_candidates}**\n\n")
        
        f.write("## 4. Expansion Analysis\n")
        f.write(f"- Old confirmed event count: {previous_confirmed}\n")
        f.write(f"- New confirmed event count: {new_confirmed}\n")
        f.write(f"- Genuinely new canonical events (from ICIMOD): {new_from_icimod}\n")
        f.write(f"- Duplicate events recovered from ICIMOD: {duplicate_icimod}\n")
        f.write(f"- New lake_uids represented: {new_lake_uids if new_lake_uids > 0 else 0}\n\n")
        
        f.write("## 5. Conclusion & Next Steps\n")
        f.write("Step 5.5 has integrated the ICIMOD database. No historical observations have been downloaded yet. ")
        if new_confirmed > 0:
            f.write("The historical evidence base has successfully expanded. The next step is to run Step 3.2 label generation and then the Step 2.5 incremental historical downloader for the new confirmed events.\n")
        else:
            f.write("Despite adding ICIMOD, no new high-confidence matches were found. Do not trigger historical downloads. The pipeline is hardened but the evidence base remains at 4 events.\n")

    print(f"Report written to {report_path.name}")

if __name__ == "__main__":
    main()
