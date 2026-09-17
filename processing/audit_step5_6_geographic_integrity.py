#!/usr/bin/env python3
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, confusion_matrix

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

def main():
    feat_path = PROCESSED_DIR / "glof_feature_matrix.parquet"
    reg_path = PROCESSED_DIR / "hma_lake_inventory_registry.parquet"
    
    df = pd.read_parquet(feat_path)
    df_lakes = pd.read_parquet(reg_path)
    
    df = pd.merge(df, df_lakes[['lake_uid', 'country']], on='lake_uid', how='left')
    
    # Validation filters
    df_lab = df[df["temporal_eligibility_status"] == "LABELABLE"].copy() if "temporal_eligibility_status" in df.columns else df.copy()
    
    # Train / Test split
    df_india = df_lab[df_lab['country'] == 'India'].copy()
    df_hma = df_lab[df_lab['country'].isin(['Nepal', 'Bhutan'])].copy()
    
    print("=== Reverse Geographic Diagnostic (Train: NP+BT, Test: IN) ===")
    if df_hma["event_within_horizon"].sum() >= 5 and df_india["event_within_horizon"].sum() >= 5:
        print("Reverse diagnostic is eligible (>=5 events in both sets).")
        
        feature_cols = [c for c in df.columns if c not in ["lake_uid", "reference_timestamp", "event_within_horizon", "temporal_eligibility_status", "country", "label_status", "horizon_days", "matched_event_id", "event_confidence", "lake_match_confidence", "processing_version", "feature_schema_version"]]
        
        X_train = df_hma[feature_cols].copy()
        y_train = df_hma["event_within_horizon"]
        X_test = df_india[feature_cols].copy()
        y_test = df_india["event_within_horizon"]
        
        valid_cols = X_train.columns[X_train.notna().any()].tolist()
        X_train = X_train[valid_cols]
        X_test = X_test[valid_cols]
        
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42))
        ])
        
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]
        
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        try:
            auc = roc_auc_score(y_test, y_prob)
        except:
            auc = np.nan
        print(f"Reverse AUC: {auc}")
        print(f"Reverse TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")
    else:
        print("INSUFFICIENT_EVENT_COUNT_FOR_REVERSE_GEOGRAPHIC_HOLDOUT")

    print("\n=== Terrain Audit ===")
    terrain_cols = ['terrain_elevation', 'terrain_slope', 'terrain_aspect']
    for c in terrain_cols:
        india_missing = df_india[c].isna().mean()
        hma_missing = df_hma[c].isna().mean()
        np_missing = df_lab[df_lab['country'] == 'Nepal'][c].isna().mean()
        bt_missing = df_lab[df_lab['country'] == 'Bhutan'][c].isna().mean()
        print(f"{c} missingness: India={india_missing*100:.1f}%, Nepal={np_missing*100:.1f}%, Bhutan={bt_missing*100:.1f}%")
        
    print("\n=== Event Level Diagnostic (Forward: Train IN -> Test NP/BT) ===")
    feature_cols = [c for c in df.columns if c not in ["lake_uid", "reference_timestamp", "event_within_horizon", "temporal_eligibility_status", "country", "label_status", "horizon_days", "matched_event_id", "event_confidence", "lake_match_confidence", "processing_version", "feature_schema_version"]]
    X_train_fwd = df_india[feature_cols].copy()
    y_train_fwd = df_india["event_within_horizon"]
    
    valid_cols_fwd = X_train_fwd.columns[X_train_fwd.notna().any()].tolist()
    X_train_fwd = X_train_fwd[valid_cols_fwd]
    
    pipe_fwd = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42))
    ])
    pipe_fwd.fit(X_train_fwd, y_train_fwd)
    
    for country in ['Nepal', 'Bhutan']:
        sub = df_lab[df_lab['country'] == country]
        X_sub = sub[valid_cols_fwd]
        y_sub = sub["event_within_horizon"]
        
        y_pred = pipe_fwd.predict(X_sub)
        tn, fp, fn, tp = confusion_matrix(y_sub, y_pred).ravel()
        print(f"{country} Total Positives: {y_sub.sum()}")
        print(f"{country} Detected (TP): {tp}")
        print(f"{country} Undetected (FN): {fn}")

if __name__ == "__main__":
    main()
