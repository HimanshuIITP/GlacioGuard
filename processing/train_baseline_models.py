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
from sklearn.model_selection import LeaveOneGroupOut

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

class NaivePrecipThreshold:
    def __init__(self, threshold=50):
        self.threshold = threshold
        self.classes_ = np.array([0, 1])
        
    def fit(self, X, y):
        return self
        
    def predict(self, X):
        return (X['precip_sum_7d'].fillna(0) > self.threshold).astype(int).values
        
    def predict_proba(self, X):
        prob = (X['precip_sum_7d'].fillna(0) > self.threshold).astype(float).values
        return np.column_stack([1 - prob, prob])

def load_config():
    with open(BASE_DIR / "config" / "model_baseline_config.yaml", "r") as f:
        return yaml.safe_load(f)

def build_model_pipeline(model_config, class_weight, feature_cols):
    model_name = model_config['name']
    
    if model_config['type'] == 'heuristic':
        return NaivePrecipThreshold(threshold=50)
        
    params = model_config['params'].copy()
    if class_weight != "None":
        params["class_weight"] = class_weight
        
    if model_config['type'] == 'sklearn.linear_model.LogisticRegression':
        clf = LogisticRegression(**params)
        return Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('clf', clf)
        ])
    elif model_config['type'] == 'sklearn.ensemble.HistGradientBoostingClassifier':
        # HistGradientBoosting handles missing values natively
        if class_weight != "None":
            params["class_weight"] = class_weight
        clf = HistGradientBoostingClassifier(**params)
        return clf
    else:
        raise ValueError(f"Unknown model type: {model_config['type']}")

def get_splits(df, split_policy, horizon):
    splits = []
    
    # LOOCV by lake_uid (which perfectly correlates with events and their controls)
    if "primary" in split_policy and split_policy["primary"]["method"] == "loocv_event":
        groups = df['lake_uid'].values
        logo = LeaveOneGroupOut()
        for i, (train_idx, test_idx) in enumerate(logo.split(df, df['event_within_horizon'], groups)):
            test_lake = df.iloc[test_idx]['lake_uid'].iloc[0]
            # Try to get the event id for naming
            test_events = df.iloc[test_idx]['matched_event_id'].dropna().unique()
            test_event = test_events[0] if len(test_events) > 0 else test_lake
            splits.append({
                'split_name': f'loocv_{test_event}',
                'train_idx': train_idx,
                'test_idx': test_idx
            })
            
    # Chronological
    if "secondary" in split_policy and split_policy["secondary"]["method"] == "chronological":
        cutoff_year = split_policy["secondary"]["cutoff_year"]
        df_years = pd.to_datetime(df['reference_timestamp']).dt.year
        train_idx = np.where(df_years <= cutoff_year)[0]
        test_idx = np.where(df_years > cutoff_year)[0]
        if len(train_idx) > 0 and len(test_idx) > 0:
            splits.append({
                'split_name': 'chronological',
                'train_idx': train_idx,
                'test_idx': test_idx
            })
            
    return splits

def main():
    print("=" * 60)
    print("  STAGE: Training Baseline Models")
    print("=" * 60)
    
    config = load_config()
    df_features = pd.read_parquet(PROCESSED_DIR / "glof_feature_matrix.parquet")
    
    predictions = []
    importances = []
    
    for horizon in config["horizons"]:
        df_h = df_features[(df_features['horizon_days'] == horizon) & 
                           (df_features['label_status'].isin(['POSITIVE', 'NEGATIVE_ELIGIBLE']))].copy()
        
        # Sort values to ensure determinism
        df_h = df_h.sort_values(by=["lake_uid", "reference_timestamp"]).reset_index(drop=True)
        
        y = df_h['event_within_horizon'].values
        
        splits = get_splits(df_h, config["split_policy"], horizon)
        
        for feature_group_name, feature_cols in config["feature_groups"].items():
            X = df_h[feature_cols]
            
            for model_cfg in config["models"]:
                for class_weight in config["class_weights"]:
                    if model_cfg['type'] == 'heuristic' and class_weight != "None":
                        continue # Weighting doesn't apply to heuristic
                        
                    for split in splits:
                        split_name = split['split_name']
                        train_idx = split['train_idx']
                        test_idx = split['test_idx']
                        
                        X_train, y_train = X.iloc[train_idx], y[train_idx]
                        X_test, y_test = X.iloc[test_idx], y[test_idx]
                        
                        pipeline = build_model_pipeline(model_cfg, class_weight, feature_cols)
                        
                        pipeline.fit(X_train, y_train)
                        
                        # Predictions for Test set
                        test_probs = pipeline.predict_proba(X_test)[:, 1]
                        test_preds = pipeline.predict(X_test)
                        
                        for i, idx in enumerate(test_idx):
                            row = df_h.iloc[idx]
                            predictions.append({
                                'lake_uid': row['lake_uid'],
                                'reference_timestamp': row['reference_timestamp'],
                                'horizon_days': horizon,
                                'model_name': model_cfg['name'],
                                'feature_group': feature_group_name,
                                'weighting_strategy': class_weight,
                                'split_name': split_name,
                                'true_label': y_test[i],
                                'predicted_probability': test_probs[i],
                                'predicted_class': test_preds[i],
                                'matched_event_id': row['matched_event_id'],
                                'label_status': row['label_status'],
                                'processing_version': '1.0'
                            })
                            
                        # Extract feature importance on train set
                        if model_cfg['type'] == 'sklearn.linear_model.LogisticRegression':
                            clf = pipeline.named_steps['clf']
                            coefs = clf.coef_[0]
                            for i, col in enumerate(feature_cols):
                                importances.append({
                                    'horizon_days': horizon,
                                    'model_name': model_cfg['name'],
                                    'feature_group': feature_group_name,
                                    'weighting_strategy': class_weight,
                                    'split_name': split_name,
                                    'feature_name': col,
                                    'importance': abs(coefs[i]),
                                    'importance_method': 'absolute_standardized_coefficient',
                                    'processing_version': '1.0'
                                })
                                
    df_preds = pd.DataFrame(predictions)
    df_preds.to_parquet(PROCESSED_DIR / "baseline_predictions.parquet", index=False)
    
    if importances:
        df_imps = pd.DataFrame(importances)
        df_imps.to_parquet(PROCESSED_DIR / "baseline_feature_importance.parquet", index=False)
        
    print(f"Generated predictions for {len(df_preds)} rows.")
    print("  [STATUS] SUCCESS")

if __name__ == "__main__":
    main()
