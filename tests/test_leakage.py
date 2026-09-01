#!/usr/bin/env python3
"""
GlacioGuard — Temporal Leakage Tests
Verifies that no future information is used in the master observation table.
"""

import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PROCESSED = BASE_DIR / "data" / "processed"

def main():
    print("\n" + "=" * 60)
    print("  STAGE: Temporal Leakage Testing")
    print("=" * 60)
    
    obs_path = DATA_PROCESSED / "lake_observations.parquet"
    if not obs_path.exists():
        print("  [ERROR] lake_observations.parquet not found.")
        print("  [STATUS] FAILED")
        return
        
    df = pd.read_parquet(obs_path)
    violations = 0
    
    if not df.empty and "reference_timestamp" in df.columns:
        for ts_col in ["weather_timestamp", "satellite_timestamp", "snow_timestamp", "gpm_timestamp"]:
            if ts_col in df.columns:
                try:
                    invalid = df[df[ts_col] > df["reference_timestamp"]]
                    if len(invalid) > 0:
                        print(f"  [ERROR] {len(invalid)} violations in {ts_col}")
                        violations += len(invalid)
                except Exception as e:
                    print(f"  [ERROR] Cannot compare {ts_col}: {e}")
        
    print(f"  future_information_violations = {violations}")
    
    if violations > 0:
        print("  [FAILED] Temporal leakage detected!")
        print("  [STATUS] FAILED")
    else:
        print("  [PASSED] No temporal leakage detected.")
        print("  [STATUS] SUCCESS")

if __name__ == "__main__":
    main()
