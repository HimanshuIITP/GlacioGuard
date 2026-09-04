#!/usr/bin/env python3
import sys
import yaml
import json
import hashlib
import argparse
from pathlib import Path
from datetime import datetime, timezone, timedelta
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "config" / "step3_2_label_config.yaml"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

def build_eligible_events():
    events_path = PROCESSED_DIR / "glof_events.parquet"
    matches_path = PROCESSED_DIR / "glof_event_lake_matches.parquet"
    
    if not events_path.exists() or not matches_path.exists():
        raise FileNotFoundError("Run Step 3.1 first. Missing events or matches.")
        
    df_events = pd.read_parquet(events_path)
    df_matches = pd.read_parquet(matches_path)
    
    # Merge matches with events
    df_full = df_events.merge(df_matches, on="event_id", how="left")
    
    records = []
    
    for idx, row in df_full.iterrows():
        # Determine eligibility
        status = "ELIGIBLE_POSITIVE"
        reason = "Valid high/medium confidence match"
        
        # Check date validity
        event_time = None
        precision = "DATETIME"
        if pd.isnull(row.get("event_date")):
            status = "EXCLUDED_INVALID_DATE"
            reason = "Missing event_date"
        else:
            try:
                date_str = str(row["event_date"])
                if len(date_str) == 10:
                    precision = "DATE"
                    # We treat date-only as 00:00:00 of that date, strictly no fabrication of 23:59:59
                    event_time = pd.to_datetime(date_str + " 00:00:00", utc=True)
                else:
                    event_time = pd.to_datetime(date_str, utc=True)
                    
                if pd.notnull(row.get("event_time")):
                    event_time = pd.to_datetime(f"{date_str} {row['event_time']}", utc=True)
                    precision = "DATETIME"
            except Exception:
                status = "EXCLUDED_INVALID_DATE"
                reason = "Unparseable date"
        
        if status == "ELIGIBLE_POSITIVE":
            if row["match_status"] == "UNMATCHED":
                status = "EXCLUDED_NO_LAKE"
                reason = "No candidate lake matched"
            elif row["match_status"] == "MANUAL_REVIEW":
                status = "EXCLUDED_MANUAL_REVIEW"
                reason = "Ambiguous lake match"
            elif row.get("event_confidence_x") not in ["HIGH", "MEDIUM"]:
                status = "EXCLUDED_LOW_EVENT_CONFIDENCE"
                reason = "Event confidence is too low"
            elif row.get("lake_match_confidence") not in ["HIGH", "MEDIUM"]:
                status = "EXCLUDED_LOW_LAKE_CONFIDENCE"
                reason = "Lake match confidence is too low"
                
        # We will save the parsed time even if excluded for logging
        r = row.to_dict()
        r["eligibility_status"] = status
        r["eligibility_reason"] = reason
        r["event_time_parsed"] = event_time
        r["event_time_precision"] = precision
        records.append(r)
        
    df_eligible = pd.DataFrame(records)
    df_eligible.to_parquet(PROCESSED_DIR / "glof_label_eligible_events.parquet", index=False)
    return df_eligible

def generate_temporal_labels(df_eligible, config):
    obs_path = PROCESSED_DIR / "lake_observations_historical.parquet"
    if not obs_path.exists():
        raise FileNotFoundError("Missing Step 2 lake_observations_historical.parquet")
        
    df_obs = pd.read_parquet(obs_path)
    
    # 1. Base candidates
    df_cand = df_obs.copy()
    df_cand["reference_timestamp"] = pd.to_datetime(df_cand["reference_timestamp"])
    
    # Spacing filtering (e.g. 24h minimum spacing per lake)
    min_spacing = timedelta(hours=config["temporal_spacing"]["min_spacing_hours"])
    df_cand = df_cand.sort_values(["lake_uid", "reference_timestamp"])
    
    spaced_rows = []
    last_times = {}
    for idx, row in df_cand.iterrows():
        luid = row["lake_uid"]
        rt = row["reference_timestamp"]
        if luid not in last_times or (rt - last_times[luid]) >= min_spacing:
            spaced_rows.append(row)
            last_times[luid] = rt
    
    df_spaced = pd.DataFrame(spaced_rows)
    total_candidates = len(df_cand)
    total_spaced = len(df_spaced)
    
    # Build list of eligible events
    pos_events = df_eligible[df_eligible["eligibility_status"] == "ELIGIBLE_POSITIVE"].copy()
    
    # Identify lakes associated with ambiguous/manual-review events
    ambiguous_lakes = set(df_eligible[df_eligible["eligibility_status"] == "EXCLUDED_MANUAL_REVIEW"]["lake_uid"].dropna())
    
    horizons = config["horizons_days"]
    pre_excl = pd.Timedelta(days=config["exclusion_window"]["pre_event_days"])
    post_excl = pd.Timedelta(days=config["exclusion_window"]["post_event_days"])
    
    labels = []
    
    for h in horizons:
        horizon_td = pd.Timedelta(days=h)
        
        for idx, row in df_spaced.iterrows():
            lake_uid = row["lake_uid"]
            t = row["reference_timestamp"]
            
            label_status = "NEGATIVE_ELIGIBLE"
            event_within = 0
            matched_evt = None
            evt_count = 0
            conf = None
            l_conf = None
            prec = None
            
            # Check for events on this lake
            lake_events = pos_events[pos_events["lake_uid"] == lake_uid]
            
            in_horizon = []
            in_contamination = False
            
            for _, evt in lake_events.iterrows():
                e_time = evt["event_time_parsed"]
                # Positive if T < E <= T + H
                if t < e_time <= t + horizon_td:
                    in_horizon.append(evt)
                else:
                    # Contamination check: if E is within [-pre_excl, +post_excl] of T
                    # T in [E - pre_excl, E + post_excl] -> T is contaminated
                    # Equivalently: E - pre_excl <= T <= E + post_excl
                    if (e_time - pre_excl) <= t <= (e_time + post_excl):
                        in_contamination = True
                        
            if in_horizon:
                label_status = "POSITIVE"
                event_within = 1
                evt_count = len(in_horizon)
                # Pick the nearest future event
                nearest = min(in_horizon, key=lambda x: x["event_time_parsed"])
                matched_evt = nearest["event_id"]
                conf = nearest["event_confidence_x"]
                l_conf = nearest["lake_match_confidence"]
                prec = nearest["event_time_precision"]
            elif in_contamination:
                label_status = "EVENT_WINDOW_EXCLUDED"
            elif lake_uid in ambiguous_lakes:
                label_status = "GEOGRAPHICALLY_UNRESOLVED"
            else:
                # Check coverage constraints based on available sensors
                w_stale = row.get("weather_stale", 9999)
                max_weather = config["coverage_thresholds"]["max_stale_hours"].get("weather", 72)
                
                # Check weather coverage
                insufficient = False
                if pd.isnull(w_stale) or w_stale > max_weather:
                    insufficient = True
                    
                if insufficient:
                    label_status = "INSUFFICIENT_FEATURE_COVERAGE"
            
            labels.append({
                "lake_uid": lake_uid,
                "reference_timestamp": t.isoformat(),
                "horizon_days": h,
                "horizon_start": t.isoformat(),
                "horizon_end": (t + horizon_td).isoformat(),
                "label_status": label_status,
                "event_within_horizon": event_within,
                "matched_event_id": matched_evt,
                "event_count_within_horizon": evt_count,
                "event_confidence": conf,
                "lake_match_confidence": l_conf,
                "event_time_precision": prec,
                "feature_window_start": (t - pd.Timedelta(days=30)).isoformat(), # default 30d window assumption for now
                "feature_window_end": t.isoformat(),
                "exclusion_status": "EXCLUDED" if label_status not in ["POSITIVE", "NEGATIVE_ELIGIBLE"] else "VALID",
                "exclusion_reason": label_status if label_status not in ["POSITIVE", "NEGATIVE_ELIGIBLE"] else None,
                "processing_version": "1.0"
            })
            
    df_labels = pd.DataFrame(labels)
    df_labels.to_parquet(PROCESSED_DIR / "glof_temporal_labels.parquet", index=False)
    
    return df_labels, total_candidates, total_spaced

def generate_controls(df_labels):
    df_pos = df_labels[df_labels["label_status"] == "POSITIVE"]
    df_neg = df_labels[df_labels["label_status"] == "NEGATIVE_ELIGIBLE"]
    
    df_neg["ref_dt"] = pd.to_datetime(df_neg["reference_timestamp"])
    
    controls = []
    
    for idx, pos in df_pos.iterrows():
        lake_uid = pos["lake_uid"]
        pos_dt = pd.to_datetime(pos["reference_timestamp"])
        pos_month = pos_dt.month
        h = pos["horizon_days"]
        
        # Candidate negatives for the same lake and horizon
        cands = df_neg[(df_neg["lake_uid"] == lake_uid) & (df_neg["horizon_days"] == h)].copy()
        
        if cands.empty:
            continue
            
        # Try to match same month
        cands["month_diff"] = abs(cands["ref_dt"].dt.month - pos_month)
        cands["month_diff"] = cands["month_diff"].apply(lambda x: min(x, 12-x)) # cyclical distance
        
        cands["temporal_distance_days"] = abs((cands["ref_dt"] - pos_dt).dt.days)
        
        # Sort by month diff (prefer same season), then by absolute distance (to get nearest valid year)
        cands = cands.sort_values(["month_diff", "temporal_distance_days"])
        
        # Pick top 3 controls
        top_k = cands.head(3)
        
        for rank, (_, cand) in enumerate(top_k.iterrows(), 1):
            ctrl_id = f"CTRL_{pos['matched_event_id']}_{cand['lake_uid']}_{cand['reference_timestamp'][:10]}_H{h}"
            ctrl_id_hash = hashlib.sha256(ctrl_id.encode()).hexdigest()[:12]
            
            controls.append({
                "control_id": "CTL_" + ctrl_id_hash,
                "positive_event_id": pos["matched_event_id"],
                "lake_uid": lake_uid,
                "control_reference_timestamp": cand["reference_timestamp"],
                "horizon_days": h,
                "control_status": "VALID",
                "matching_month": cand["ref_dt"].month,
                "matching_year": cand["ref_dt"].year,
                "temporal_distance_days": cand["temporal_distance_days"],
                "selection_rank": rank,
                "selection_reason": "Same-lake month-matched nearest-year",
                "processing_version": "1.0"
            })
            
    df_controls = pd.DataFrame(controls)
    if not df_controls.empty:
        df_controls.to_parquet(PROCESSED_DIR / "glof_control_periods.parquet", index=False)
    else:
        pd.DataFrame(columns=["control_id", "positive_event_id"]).to_parquet(PROCESSED_DIR / "glof_control_periods.parquet", index=False)
        
    return df_controls

def check_leakage(df_labels):
    # Leakage test: future features should never have timestamp > reference_timestamp
    obs_path = PROCESSED_DIR / "lake_observations_historical.parquet"
    if not obs_path.exists():
        return 0
        
    df_obs = pd.read_parquet(obs_path)
    # We check if any row in df_obs has a feature timestamp (e.g. satellite_timestamp) > reference_timestamp
    violations = 0
    if "satellite_timestamp" in df_obs.columns:
        mask = pd.to_datetime(df_obs["satellite_timestamp"]) > pd.to_datetime(df_obs["reference_timestamp"])
        violations += mask.sum()
    if "weather_timestamp" in df_obs.columns:
        mask = pd.to_datetime(df_obs["weather_timestamp"]) > pd.to_datetime(df_obs["reference_timestamp"])
        violations += mask.sum()
    return violations

def write_report(df_eligible, df_labels, df_controls, cand_stats):
    report_path = PROCESSED_DIR / "step3_2_quality_report.md"
    
    with open(report_path, "w") as f:
        f.write("# Step 3.2 Label Quality Report\n\n")
        
        f.write("## Event Eligibility\n")
        counts = df_eligible["eligibility_status"].value_counts()
        for k, v in counts.items():
            f.write(f"- {k}: {v}\n")
            
        f.write("\n## Label Distribution\n")
        f.write(f"- Total candidate rows before spacing: {cand_stats[0]}\n")
        f.write(f"- Total candidate rows after 24h spacing: {cand_stats[1]}\n\n")
        
        for h in df_labels["horizon_days"].unique():
            f.write(f"### {h}-Day Horizon\n")
            sub = df_labels[df_labels["horizon_days"] == h]
            lcounts = sub["label_status"].value_counts()
            for k, v in lcounts.items():
                f.write(f"- {k}: {v}\n")
            f.write("\n")
            
        f.write("## Control Periods\n")
        f.write(f"- Total controls generated: {len(df_controls)}\n")
        if not df_controls.empty:
            pos_with_ctrl = df_controls["positive_event_id"].nunique()
            f.write(f"- Positive events with >=1 control: {pos_with_ctrl}\n")
            
        f.write("\n## Event-Level Validation\n")
        pos_events = df_eligible[df_eligible["eligibility_status"] == "ELIGIBLE_POSITIVE"]
        for _, evt in pos_events.iterrows():
            f.write(f"### Event: {evt['event_id']} (Lake: {evt['lake_uid']}, Date: {evt['event_date']})\n")
            f.write(f"- event_time_precision: {evt['event_time_precision']}\n")
            for h in df_labels["horizon_days"].unique():
                sub = df_labels[(df_labels["matched_event_id"] == evt["event_id"]) & (df_labels["horizon_days"] == h) & (df_labels["label_status"] == "POSITIVE")]
                has_pos = len(sub) > 0
                f.write(f"- POSITIVE at {h}d: {'YES' if has_pos else 'NO'} ({len(sub)} rows)\n")
                if has_pos:
                    min_dt = sub["reference_timestamp"].min()
                    max_dt = sub["reference_timestamp"].max()
                    f.write(f"  - Earliest positive reference: {min_dt}\n")
                    f.write(f"  - Latest positive reference: {max_dt}\n")
            f.write("\n")

        f.write("## Leakage Check\n")
        v = check_leakage(df_labels)
        f.write(f"- future_feature_timestamp > reference_timestamp: {v} violations\n")
        f.write(f"- Leakage test: {'PASS' if v == 0 else 'FAIL'}\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("  STAGE: Temporal Label Generation")
    print("=" * 60)
    
    config = load_config()
    df_eligible = build_eligible_events()
    df_labels, tot_cand, tot_spaced = generate_temporal_labels(df_eligible, config)
    df_controls = generate_controls(df_labels)
    
    violations = check_leakage(df_labels)
    if violations > 0:
        print(f"  [ERROR] Leakage test failed with {violations} violations!")
        sys.exit(1)
        
    write_report(df_eligible, df_labels, df_controls, (tot_cand, tot_spaced))
    print("  Outputs generated. Zero leakage violations.")
    print("  [STATUS] SUCCESS")

if __name__ == "__main__":
    main()
