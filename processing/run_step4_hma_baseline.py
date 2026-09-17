#!/usr/bin/env python3
import sys
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score, confusion_matrix, balanced_accuracy_score

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

class NaivePrecipThreshold:
    def __init__(self, threshold=50):
        self.threshold = threshold
        
    def fit(self, X, y=None):
        return self
        
    def predict(self, X):
        precip = X.get('precipitation_mm_7d_sum', pd.Series(np.zeros(len(X))))
        return (precip > self.threshold).astype(int)
        
    def predict_proba(self, X):
        precip = X.get('precipitation_mm_7d_sum', pd.Series(np.zeros(len(X))))
        # return a pseudo-probability between 0 and 1
        prob = np.clip(precip / (self.threshold * 2), 0, 1)
        # sklearn expects Nx2
        return np.vstack([1 - prob, prob]).T

def get_metrics(y_true, y_pred, y_prob):
    if len(np.unique(y_true)) < 2:
        return {
            "ROC_AUC": np.nan, "PR_AUC": np.nan, "Precision": np.nan,
            "Recall": np.nan, "F1": np.nan, "Balanced_Acc": np.nan, "TP": np.nan, "FP": np.nan, "FN": np.nan, "TN": np.nan
        }
        
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "ROC_AUC": roc_auc_score(y_true, y_prob),
        "PR_AUC": average_precision_score(y_true, y_prob),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "Balanced_Acc": balanced_accuracy_score(y_true, y_pred),
        "TP": tp, "FP": fp, "FN": fn, "TN": tn
    }

def main():
    print("=" * 60)
    print("  STAGE: Step 4 HMA Geographic Baseline Rerun")
    print("=" * 60)
    
    feat_path = PROCESSED_DIR / "glof_feature_matrix.parquet"
    if not feat_path.exists():
        print(f"Missing {feat_path.name}")
        sys.exit(1)
        
    df = pd.read_parquet(feat_path)
    print(f"Loaded {len(df)} feature rows.")
    
    # Load lake inventory to get country labels
    reg_path = PROCESSED_DIR / "hma_lake_inventory_registry.parquet"
    if not reg_path.exists():
        # Fallback
        reg_path = PROCESSED_DIR / "india_glacial_lakes_2022.parquet"
    
    df_lakes = pd.read_parquet(reg_path)
    
    if 'country' not in df_lakes.columns:
        print("[ERROR] No country column in registry.")
        sys.exit(1)
        
    df = pd.merge(df, df_lakes[['lake_uid', 'country']], on='lake_uid', how='left')
    
    # Validation filters
    if 'event_id' in df.columns:
        print("[ERROR] Leakage: 'event_id' found in feature matrix.")
        sys.exit(1)
        
    if df.empty:
        print("Feature matrix is empty.")
        sys.exit(1)
        
    # We evaluate Geographic Holdout: Train on India, Test on Nepal+Bhutan
    df_train = df[df['country'] == 'India'].copy()
    df_test = df[df['country'].isin(['Nepal', 'Bhutan'])].copy()
    
    # Verify Operational Threshold >= 5 positive events in each set
    train_pos_events = df_train[df_train["event_within_horizon"] == 1]["lake_uid"].nunique()
    test_pos_events = df_test[df_test["event_within_horizon"] == 1]["lake_uid"].nunique()
    
    print("\n--- Geographic Diagnostic Integrity Checks ---")
    print(f"India (Train)  - Total rows: {len(df_train)}, Positive events: {train_pos_events}")
    print(f"Nepal+Bhutan (Test) - Total rows: {len(df_test)}, Positive events: {test_pos_events}")
    
    if train_pos_events < 5:
        print("\n[WARNING] INSUFFICIENT_EVENT_COUNT_FOR_GEOGRAPHIC_HOLDOUT: Train set has < 5 positive events.")
    if test_pos_events < 5:
        print("\n[WARNING] INSUFFICIENT_EVENT_COUNT_FOR_GEOGRAPHIC_HOLDOUT: Test set has < 5 positive events.")
        
    # Group configurations
    feature_groups = {
        "Weather_Only": [c for c in df.columns if c.startswith("temperature") or c.startswith("precipitation")],
        "Weather_Availability": [c for c in df.columns if c.startswith("temperature") or c.startswith("precipitation") or "unavailable" in c],
        "All_Available": [c for c in df.columns if c not in ["lake_uid", "reference_timestamp", "event_within_horizon", "temporal_eligibility_status", "country", "label_status", "horizon_days", "matched_event_id", "event_confidence", "lake_match_confidence", "processing_version", "feature_schema_version"]]
    }
    
    models = {
        "LogisticRegression": Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42))
        ]),
        "HistGradientBoosting": HistGradientBoostingClassifier(class_weight="balanced", random_state=42),
        "NaivePrecipThreshold": NaivePrecipThreshold(threshold=50)
    }
    
    results = []
    
    for fg_name, f_cols in feature_groups.items():
        # Ensure cols exist
        f_cols = [c for c in f_cols if c in df.columns]
        
        X_train = df_train[f_cols]
        y_train = df_train["event_within_horizon"]
        X_test = df_test[f_cols]
        y_test = df_test["event_within_horizon"]
        
        for m_name, model in models.items():
            if m_name == "NaivePrecipThreshold" and "precipitation_mm_7d_sum" not in f_cols:
                continue
                
            try:
                # Naive model doesn't need to fit, but we call it anyway
                if m_name == "NaivePrecipThreshold":
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    y_prob = model.predict_proba(X_test)[:, 1]
                else:
                    if y_train.nunique() < 2:
                        continue # Cannot train
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    y_prob = model.predict_proba(X_test)[:, 1]
                    
                metrics = get_metrics(y_test, y_pred, y_prob)
                metrics["Feature_Group"] = fg_name
                metrics["Model"] = m_name
                results.append(metrics)
                
            except Exception as e:
                print(f"Error training {m_name} on {fg_name}: {e}")
                
    df_res = pd.DataFrame(results)
    if not df_res.empty:
        cols = ["Model", "Feature_Group"] + [c for c in df_res.columns if c not in ["Model", "Feature_Group"]]
        df_res = df_res[cols]
        
        report_path = PROCESSED_DIR / "step4_hma_geographic_baseline_report.md"
        with open(report_path, "w") as f:
            f.write("# Step 4 HMA Geographic Baseline Report\n\n")
            f.write(f"**Train**: India ({train_pos_events} positive events)\n")
            f.write(f"**Test**: Nepal + Bhutan ({test_pos_events} positive events)\n\n")
            f.write(df_res.to_csv(sep='|', index=False))
            
        print(f"\nSaved {report_path.name}")
        print("\n" + df_res.to_csv(sep='|', index=False))
    else:
        print("\nNo results generated. Likely insufficient labels.")
        
    print("\n  [STATUS] SUCCESS")

if __name__ == "__main__":
    main()
