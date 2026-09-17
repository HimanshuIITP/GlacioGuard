import sys
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

def main():
    print("=" * 60)
    print("  STAGE: Audit HMA Temporal Eligibility")
    print("=" * 60)
    
    events_df = pd.read_parquet(PROCESSED_DIR / "hma_event_inventory.parquet")
    matches_df = pd.read_parquet(PROCESSED_DIR / "hma_event_lake_matches.parquet")
    
    # Merge event data with matches
    df = pd.merge(events_df, matches_df, on='event_id')
    
    audit_results = []
    
    for idx, row in df.iterrows():
        status = 'TEMPORALLY_UNLABELABLE'
        reason = 'No valid event date or unknown precision'
        
        # Only HIGH_CONFIDENCE_MATCH events are eligible for labels
        # Wait, the user specifically requested:
        # "Primary positive eligibility remains: event_confidence IN {HIGH, MEDIUM} AND lake_match_status == HIGH_CONFIDENCE_MATCH AND valid event_date exists"
        
        if row['match_status'] != 'HIGH_CONFIDENCE_MATCH':
            status = 'TEMPORALLY_UNLABELABLE'
            reason = f"spatial_match_status is {row['match_status']}, not HIGH_CONFIDENCE_MATCH"
        elif row['event_confidence'] not in ['HIGH', 'MEDIUM']:
            status = 'TEMPORALLY_UNLABELABLE'
            reason = f"event_confidence is {row['event_confidence']}, not HIGH/MEDIUM"
        elif pd.isna(row['event_date']):
            status = 'INVALID_DATE'
            reason = 'event_date is missing entirely'
        elif row['event_time_precision'] not in ['DATE', 'DATETIME']:
            status = 'TEMPORALLY_UNLABELABLE'
            reason = f"event_time_precision is {row['event_time_precision']}, which is insufficiently exact"
        else:
            status = 'LABELABLE'
            reason = 'Valid precise date and HIGH_CONFIDENCE_MATCH'
            
        audit_results.append({
            'event_id': row['event_id'],
            'lake_uid': row['lake_uid'],
            'event_date': row['event_date'],
            'event_time': row['event_time'],
            'spatial_match_status': row['match_status'],
            'temporal_eligibility_status': status,
            'reason': reason,
            'processing_version': '1.0'
        })
        
    df_audit = pd.DataFrame(audit_results)
    
    out_path = PROCESSED_DIR / "hma_temporal_event_eligibility.parquet"
    df_audit.to_parquet(out_path, index=False)
    
    print("Audit Results:")
    print(df_audit['temporal_eligibility_status'].value_counts())
    print(f"\nSaved to {out_path.name}")
    
    # Also calculate the actual usable POSITIVE event counts (LABELABLE)
    df_labels = df_audit[df_audit['temporal_eligibility_status'] == 'LABELABLE']
    # Join country
    df_labels = pd.merge(df_labels, events_df[['event_id', 'country']], on='event_id')
    print("\nEligible Positive Events by Country:")
    print(df_labels['country'].value_counts())
    
if __name__ == "__main__":
    main()
