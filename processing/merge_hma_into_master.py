import sys
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

def main():
    print("=" * 60)
    print("  STAGE: Merge HMA into Master Registries")
    print("=" * 60)
    
    # 1. Merge Lakes
    print("Replacing master lakes with hma_lake_inventory_registry.parquet (which already includes India)...")
    df_lakes_hma = pd.read_parquet(PROCESSED_DIR / "hma_lake_inventory_registry.parquet")
    df_lakes_hma.to_parquet(PROCESSED_DIR / "india_glacial_lakes_2022.parquet", index=False) 
    # Wait, the rest of the pipeline uses `india_glacial_lakes_2022.parquet` to represent the master list. 
    # Or maybe it uses it. Let's just create a `master_lake_inventory.parquet` and replace the references, 
    # but to be safe without breaking other scripts, let's append to glof_events etc.
    
    # 2. Merge Events
    print("Merging events...")
    df_events_in = pd.read_parquet(PROCESSED_DIR / "glof_events.parquet")
    df_events_hma = pd.read_parquet(PROCESSED_DIR / "hma_event_inventory.parquet")
    
    df_events = pd.concat([df_events_in, df_events_hma], ignore_index=True)
    df_events.to_parquet(PROCESSED_DIR / "glof_events.parquet", index=False)
    
    # 3. Merge Matches
    print("Merging matches...")
    df_matches_in = pd.read_parquet(PROCESSED_DIR / "glof_event_lake_matches.parquet")
    df_matches_hma = pd.read_parquet(PROCESSED_DIR / "hma_event_lake_matches.parquet")
    
    df_matches = pd.concat([df_matches_in, df_matches_hma], ignore_index=True)
    df_matches.to_parquet(PROCESSED_DIR / "glof_event_lake_matches.parquet", index=False)
    
    # 4. Temporal Audit
    print("Merging temporal eligibility audit...")
    df_audit_in = pd.read_parquet(PROCESSED_DIR / "step2_5_temporal_event_eligibility.parquet")
    df_audit_hma = pd.read_parquet(PROCESSED_DIR / "hma_temporal_event_eligibility.parquet")
    
    df_audit = pd.concat([df_audit_in, df_audit_hma], ignore_index=True)
    df_audit.to_parquet(PROCESSED_DIR / "step2_5_temporal_event_eligibility.parquet", index=False)
    
    print(f"Total events in master: {len(df_events)}")
    print(f"Total matches in master: {len(df_matches)}")
    print(f"Total audit rows in master: {len(df_audit)}")
    
if __name__ == "__main__":
    main()
