#!/usr/bin/env python3
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

def run_script(script_name):
    script_path = BASE_DIR / "processing" / script_name
    print(f"\n>>> Running {script_name}")
    result = subprocess.run([sys.executable, str(script_path)])
    if result.returncode != 0:
        print(f"Error executing {script_name}")
        sys.exit(result.returncode)

def main():
    print("============================================================")
    print("  GlacioGuard Step 4: Baseline Models")
    print("============================================================")
    
    run_script("train_baseline_models.py")
    run_script("evaluate_baseline_models.py")
    
    # Idempotency check
    print("\n--- Running Idempotency Check ---")
    run_script("train_baseline_models.py")
    run_script("evaluate_baseline_models.py")
    
    print("\n[STATUS] Step 4 Completed Successfully")

if __name__ == "__main__":
    main()
