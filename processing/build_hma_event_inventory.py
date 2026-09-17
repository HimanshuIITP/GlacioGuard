import sys
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone

BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = BASE_DIR / "data" / "raw" / "glof_events" / "icimod" / "fidelsteiner-HMAGLOFDB-1d975de" / "Database" / "GLOFs" / "HMAGLOFDB.csv"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

def main():
    print("=" * 60)
    print("  STAGE: Build HMA Event Inventory")
    print("=" * 60)
    
    if not CSV_PATH.exists():
        print(f"[ERROR] Could not find HMA GLOF DB at {CSV_PATH}")
        sys.exit(1)
            
    print(f"Loading HMA GLOF DB from {CSV_PATH.name}...")
    df = pd.read_csv(CSV_PATH, encoding='latin1')
        
    print(f"Raw HMA GLOF records: {len(df)}")
    
    # Standardize country mapping to match our inventories
    df['canonical_country'] = df['Country'].replace({
        'NP': 'Nepal',
        'Nepal': 'Nepal',
        'BT': 'Bhutan',
        'Bhutan': 'Bhutan'
    })
        
    df_nb = df[df['canonical_country'].isin(['Nepal', 'Bhutan'])].copy()
    print(f"Filtered down to {len(df_nb)} events for Nepal and Bhutan.")
    
    events = []
    for idx, row in df_nb.iterrows():
        event_id = str(row.get('GF_ID', f"HMAGLOF_{idx}"))
            
        y_exact = row.get('Year_exact')
        y_approx = row.get('Year_approx')
        m = row.get('Month')
        d = row.get('Day')
        
        y = y_exact if pd.notna(y_exact) else y_approx
        
        event_date = None
        event_time_precision = 'UNKNOWN'
        
        if pd.notna(y) and pd.notna(m) and pd.notna(d):
            try:
                event_date = f"{int(y):04d}-{int(m):02d}-{int(d):02d}"
                event_time_precision = 'DATE'
            except ValueError:
                pass
        elif pd.notna(y) and pd.notna(m):
            try:
                event_date = f"{int(y):04d}-{int(m):02d}-01"
                event_time_precision = 'MONTH'
            except ValueError:
                pass
        elif pd.notna(y):
            try:
                event_date = f"{int(y):04d}-01-01"
                event_time_precision = 'YEAR'
            except ValueError:
                pass
                
        events.append({
            'event_id': f"ICIMOD_{event_id}",
            'event_date': event_date,
            'event_time': None,
            'event_time_precision': event_time_precision,
            'latitude': row.get('Lat_lake'),
            'longitude': row.get('Lon_lake'),
            'country': row['canonical_country'],
            'region': row.get('Province', None),
            'district': None,
            'lake_name': row.get('Lake_name', None),
            'reported_lake_id': row.get('GL_ID', None),
            'trigger': row.get('Driver_GLOF', row.get('Mechanism', None)),
            'description': row.get('Impact', None),
            'fatalities': row.get('Lives_total', None),
            'source': 'ICIMOD HMA GLOF DB',
            'source_url': None,
            'source_publication': row.get('Ref_scientific_full', 'Fidelsteiner et al.'),
            'event_confidence': 'HIGH',
            'notes': str(row.get('Remarks', '')),
            'retrieved_at': datetime.now(timezone.utc).isoformat(),
            'processing_version': '1.0'
        })
        
    df_events = pd.DataFrame(events)
    
    labelable = df_events[df_events['event_time_precision'].isin(['DATE', 'DATETIME'])]
    print(f"Events with usable exact dates (DATE/DATETIME precision): {len(labelable)}")
    print(f"Events with unlabelable dates (YEAR/MONTH precision): {len(df_events) - len(labelable)}")
    
    out_path = PROCESSED_DIR / "hma_event_inventory.parquet"
    df_events.to_parquet(out_path, index=False)
    print(f"Saved normalized event inventory to {out_path.name}")

if __name__ == "__main__":
    main()
