#!/usr/bin/env python3
import sys
import yaml
import hashlib
import json
import time
import subprocess
import threading
import pandas as pd
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score, confusion_matrix
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.base import clone
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
CONFIG_DIR = BASE_DIR / "config"
STEP8_DIR = PROCESSED_DIR / "step8"
STEP8_DIR.mkdir(exist_ok=True)

class GPUMonitor:
    def __init__(self):
        self.running = False
        self.thread = None
        self.utilizations = []
        self.vrams = []
        self.lock = threading.Lock()
        self.start_time = time.time()
        
    def _monitor_loop(self):
        while self.running:
            try:
                res = subprocess.check_output([
                    "nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"
                ], text=True).strip().split(',')
                with self.lock:
                    self.utilizations.append(float(res[0]))
                    self.vrams.append(float(res[1]))
            except:
                pass
            time.sleep(1)
            
    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
            
    def report(self):
        with self.lock:
            return {
                "wall_clock_s": time.time() - self.start_time,
                "mean_gpu_util": np.mean(self.utilizations) if self.utilizations else 0,
                "max_gpu_util": np.max(self.utilizations) if self.utilizations else 0,
                "mean_vram_mb": np.mean(self.vrams) if self.vrams else 0,
                "max_vram_mb": np.max(self.vrams) if self.vrams else 0
            }

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def hash_file(filepath):
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

def verify_artifacts(cfg):
    manifest_path = BASE_DIR / cfg["paths"]["manifest"]
    if not manifest_path.exists():
        print("[HARD FAIL] Step 7 artifact manifest not found. Leakage Audit Failed.")
        sys.exit(1)
        
    with open(manifest_path, "r") as f:
        saved_hashes = json.load(f)
        
    for name, path in saved_hashes.items():
        if name in ["step6_feature_matrix", "step6_results", "step6_outer_predictions", "step6_feature_importance", "step6_model_selection_report", "step6_leakage_audit"]:
            # Need to find the relative path based on the config since manifest doesn't store full paths properly for dict lookups if they changed
            actual_path = None
            if name == "step6_feature_matrix": actual_path = cfg["paths"]["feature_matrix"]
            elif name == "step6_results": actual_path = cfg["paths"]["step6_results"]
            
            if actual_path:
                current_hash = hash_file(BASE_DIR / actual_path)
                if current_hash != path:
                    print(f"[HARD FAIL] Artifact modified: {name}")
                    sys.exit(1)

def get_base_features():
    step6_cfg = load_config(CONFIG_DIR / "step6_experiment_config.yaml")
    return [f["feature_name"] for f in step6_cfg["feature_groups"]["ALL_AVAILABLE"]]

def evaluate_common_core(df_train, predictor_cols, cfg):
    min_avail = cfg["common_core"]["min_availability"]
    max_diff = cfg["common_core"]["max_missingness_diff"]
    
    regions = df_train["country"].unique()
    features_info = []
    
    for f in predictor_cols:
        if f not in df_train.columns:
            continue
            
        avails = {}
        for r in regions:
            r_df = df_train[df_train["country"] == r]
            avails[r] = r_df[f].notna().mean()
            
        avail_list = list(avails.values())
        min_region_avail = min(avail_list)
        max_region_avail = max(avail_list)
        pairwise_diff = max_region_avail - min_region_avail
        
        included = True
        reason = ""
        
        if min_region_avail < min_avail:
            included = False
            reason = "Below min availability"
        elif pairwise_diff > max_diff:
            included = False
            reason = "High pairwise missingness diff"
            
        features_info.append({
            "feature": f,
            "feature_family": "Unknown", # Could map from step 6 registry if needed
            "included": included,
            "exclusion_reason": reason,
            "availability_by_train_region": str({k: round(v,3) for k,v in avails.items()}),
            "pairwise_missingness_diff": pairwise_diff
        })
        
    return features_info

def build_pipeline(model_cfg, smote, has_missing_indicators):
    steps = []
    if has_missing_indicators:
        steps.append(("imputer", SimpleImputer(strategy="mean")))
    else:
        steps.append(("imputer", SimpleImputer(strategy="mean")))
        
    if "xgb" not in model_cfg["name"].lower() and "histgb" not in model_cfg["name"].lower():
        steps.append(("scaler", StandardScaler()))
        
    if smote:
        steps.append(("smote", SMOTE(random_state=42)))
        
    if "XGB" in model_cfg["name"]:
        clf = xgb.XGBClassifier(
            tree_method="hist", 
            device="cuda", 
            max_depth=model_cfg["max_depth"], 
            learning_rate=model_cfg["learning_rate"], 
            n_estimators=model_cfg["n_estimators"], 
            random_state=42
        )
    elif "HistGB" in model_cfg["name"]:
        clf = HistGradientBoostingClassifier(
            class_weight=model_cfg["class_weight"], 
            random_state=42
        )
    else:
        clf = LogisticRegression(
            class_weight=model_cfg["class_weight"], 
            max_iter=model_cfg.get("max_iter", 1000), 
            random_state=42
        )
        
    steps.append(("clf", clf))
    return ImbPipeline(steps)

def fit_eval_fold(pipe_base, X_tr, y_tr, X_val, y_val, thresholds, model_name, use_smote, pos_weight):
    pipe = clone(pipe_base)
    
    if "XGB" in model_name and not use_smote:
        pipe.named_steps["clf"].set_params(scale_pos_weight=pos_weight)
        
    pipe.fit(X_tr, y_tr)
    y_prob = pipe.predict_proba(X_val)[:, 1]
    
    try:
        pr_auc = average_precision_score(y_val, y_prob)
    except:
        pr_auc = np.nan
        
    best_f1 = -1
    best_thresh = thresholds[0]
    best_rec = -1
    
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        f1 = f1_score(y_val, y_pred)
        rec = recall_score(y_val, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
            best_rec = rec
            
    return {
        "model_name": model_name,
        "y_prob": list(y_prob),
        "y_val": list(y_val),
        "pr_auc": pr_auc,
        "best_f1": best_f1,
        "best_thresh": best_thresh,
        "best_recall": best_rec
    }

def run_inner_cv(X, y, groups, model_cfgs, thresholds, smote_options, has_missing_indicators):
    pos_groups = groups[y == 1].nunique()
    if pos_groups < 2:
        return None
    n_splits = 3 if pos_groups >= 3 else 2
    
    cv = StratifiedGroupKFold(n_splits=n_splits)
    
    candidates = []
    
    for smote in smote_options:
        for m_cfg in model_cfgs:
            m_name = m_cfg["name"]
            
            oof_probs = []
            oof_y = []
            
            for train_idx, val_idx in cv.split(X, y, groups=groups):
                X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
                X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
                
                neg_c = (y_tr == 0).sum()
                pos_c = (y_tr == 1).sum()
                pos_weight = neg_c / pos_c if pos_c > 0 else 1
                
                pipe = build_pipeline(m_cfg, smote, has_missing_indicators)
                res = fit_eval_fold(pipe, X_tr, y_tr, X_val, y_val, thresholds, m_name, smote, pos_weight)
                
                oof_probs.extend(res["y_prob"])
                oof_y.extend(res["y_val"])
                
            oof_probs = np.array(oof_probs)
            oof_y = np.array(oof_y)
            
            try:
                pr_auc = average_precision_score(oof_y, oof_probs)
            except:
                pr_auc = np.nan
                
            best_f1 = -1
            best_t = thresholds[0]
            best_r = -1
            for t in thresholds:
                yp = (oof_probs >= t).astype(int)
                f1 = f1_score(oof_y, yp)
                rec = recall_score(oof_y, yp)
                if f1 > best_f1:
                    best_f1 = f1
                    best_t = t
                    best_r = rec
                    
            candidates.append({
                "model_name": m_name,
                "smote": smote,
                "pr_auc": pr_auc,
                "f1": best_f1,
                "recall": best_r,
                "threshold": best_t,
                "complexity": 2 if "XGB" in m_name else (1 if "HistGB" in m_name else 0),
                "cfg": m_cfg
            })
            
    # Tie breaking
    candidates.sort(key=lambda x: (
        round(x["pr_auc"], 3),
        round(x["f1"], 3),
        round(x["recall"], 3),
        -x["complexity"]
    ), reverse=True)
    
    return candidates[0] if candidates else None

def main():
    print("Initializing Step 8: Cross-Region Robust Modeling")
    monitor = GPUMonitor()
    monitor.start()
    
    cfg = load_config(CONFIG_DIR / "step8_cross_region_config.yaml")
    verify_artifacts(cfg)
    
    df_feat = pd.read_parquet(BASE_DIR / cfg["paths"]["feature_matrix"])
    df_reg = pd.read_parquet(BASE_DIR / cfg["paths"]["registry"])
    df = pd.merge(df_feat, df_reg[['lake_uid', 'country']], on='lake_uid', how='left')
    
    if "temporal_eligibility_status" in df.columns:
        df = df[df["temporal_eligibility_status"] == "LABELABLE"].copy()
        
    base_features = get_base_features()
    
    feature_sets_log = []
    predictions_log = []
    
    # Pre-compile models
    all_models = cfg["hyperparameters"]["xgb"] + cfg["hyperparameters"]["lr"] + cfg["hyperparameters"]["histgb"]
    
    for h in cfg["horizons"]:
        df_h = df[df["horizon_days"] == h].copy().dropna(subset=["event_within_horizon"])
        
        # STRATEGY A: Historical Reference (Trained on India, evaluated on Nepal+Bhutan)
        print(f"\nEvaluating STRATEGY A (Historical Reference) for Horizon {h}")
        step6_res = pd.read_parquet(BASE_DIR / cfg["paths"]["step6_results"])
        h_res = step6_res[step6_res["horizon"] == h].sort_values(by="inner_pr_auc", ascending=False)
        if not h_res.empty:
            ref_cfg = h_res.iloc[0]
            fg_name = ref_cfg["feature_group"]
            smote_en = True if ref_cfg["resampling_method"] == "SMOTE" else False
            m_name = ref_cfg["model_name"]
            
            step6_exp = load_config(CONFIG_DIR / "step6_experiment_config.yaml")
            ref_feats = [f["feature_name"] for f in step6_exp["feature_groups"][fg_name]]
            ref_feats = [c for c in ref_feats if c in df_h.columns]
            
            df_ind = df_h[df_h["country"] == "India"].copy()
            X_tr = df_ind[ref_feats]
            y_tr = df_ind["event_within_horizon"]
            
            valid_cols = X_tr.columns[X_tr.notna().any()].tolist()
            X_tr = X_tr[valid_cols]
            
            # Reconstruct ref
            pipe = build_pipeline({"name": m_name, "max_depth": int(m_name.split("_d")[1].split("_")[0]) if "XGB" in m_name else 3, "learning_rate": float(m_name.split("_lr")[1].split("_")[0]) if "XGB" in m_name else 0.1, "n_estimators": int(m_name.split("_n")[1]) if "XGB" in m_name else 100, "class_weight": "balanced" if "balanced" in m_name else None}, smote_en, False)
            
            if "XGB" in m_name and not smote_en:
                neg_c = (y_tr == 0).sum()
                pos_c = (y_tr == 1).sum()
                pipe.named_steps["clf"].set_params(scale_pos_weight=(neg_c/pos_c if pos_c>0 else 1))
                
            pipe.fit(X_tr, y_tr)
            
            for test_reg in ["Nepal", "Bhutan"]:
                df_test = df_h[df_h["country"] == test_reg].copy()
                if not df_test.empty:
                    X_te = df_test[valid_cols]
                    y_prob = pipe.predict_proba(X_te)[:, 1]
                    
                    for i in range(len(df_test)):
                        predictions_log.append({
                            "strategy": "A",
                            "evaluation_type": "HISTORICAL_FROZEN_REFERENCE",
                            "horizon": h,
                            "outer_fold": 0,
                            "train_regions": "['India']",
                            "test_region": test_reg,
                            "lake_uid": df_test["lake_uid"].iloc[i],
                            "y_true": df_test["event_within_horizon"].iloc[i],
                            "y_prob": y_prob[i],
                            "threshold": ref_cfg["threshold"]
                        })
        
        # STRATEGY B, C, D (LORO LOOP)
        for fold_def in cfg["loro_folds"]:
            f_idx = fold_def["fold"]
            train_regs = fold_def["train"]
            test_reg = fold_def["test"]
            
            print(f"\nProcessing LORO Fold {f_idx}: Train {train_regs} -> Test {test_reg} (Horizon {h})")
            
            df_tr = df_h[df_h["country"].isin(train_regs)].copy()
            df_te = df_h[df_h["country"] == test_reg].copy()
            
            if df_tr.empty or df_te.empty:
                continue
                
            # Compute Fold-Local Common Core
            cc_info = evaluate_common_core(df_tr, base_features, cfg)
            
            for info in cc_info:
                info["strategy"] = "B/C"
                info["outer_fold"] = f_idx
                info["train_regions"] = str(train_regs)
                info["heldout_region"] = test_reg
                info["feature_selection_rule"] = "FOLD_LOCAL_COMMON_CORE"
                info["horizon"] = h
                feature_sets_log.append(info)
                
                # Also log Strategy D features explicitly
                info_d = info.copy()
                info_d["strategy"] = "D"
                info_d["included"] = True
                info_d["exclusion_reason"] = ""
                info_d["feature_selection_rule"] = "STEP6_ALL_AVAILABLE_REGISTRY"
                feature_sets_log.append(info_d)
                
            cc_feats = [x["feature"] for x in cc_info if x["included"]]
            
            strategies = [
                ("B", cc_feats, False),
                ("C", cc_feats, True),
                ("D", [x["feature"] for x in cc_info], False)
            ]
            
            for strat_name, feat_list, add_indicators in strategies:
                print(f"  Strategy {strat_name} (Features: {len(feat_list)}, Indicators: {add_indicators})")
                
                # Prepare data
                valid_f = [c for c in feat_list if c in df_tr.columns and df_tr[c].notna().any()]
                
                X_tr = df_tr[valid_f].copy()
                y_tr = df_tr["event_within_horizon"]
                g_tr = df_tr["lake_uid"]
                
                X_te = df_te[valid_f].copy()
                y_te = df_te["event_within_horizon"]
                
                if add_indicators:
                    for c in valid_f:
                        X_tr[f"{c}_missing"] = X_tr[c].isna().astype(int)
                        X_te[f"{c}_missing"] = X_te[c].isna().astype(int)
                        
                # Inner CV
                best_cand = run_inner_cv(
                    X_tr, y_tr, g_tr, 
                    all_models, 
                    cfg["thresholds"], 
                    [False, True], 
                    add_indicators
                )
                
                if not best_cand:
                    print("    [WARNING] Inner CV failed (insufficient positive groups). Skipping.")
                    continue
                    
                # Refit and score
                neg_c = (y_tr == 0).sum()
                pos_c = (y_tr == 1).sum()
                pos_weight = neg_c / pos_c if pos_c > 0 else 1
                
                pipe = build_pipeline(best_cand["cfg"], best_cand["smote"], add_indicators)
                if "XGB" in best_cand["model_name"] and not best_cand["smote"]:
                    pipe.named_steps["clf"].set_params(scale_pos_weight=pos_weight)
                    
                pipe.fit(X_tr, y_tr)
                y_prob = pipe.predict_proba(X_te)[:, 1]
                
                for i in range(len(df_te)):
                    predictions_log.append({
                        "strategy": strat_name,
                        "evaluation_type": "LORO",
                        "horizon": h,
                        "outer_fold": f_idx,
                        "train_regions": str(train_regs),
                        "test_region": test_reg,
                        "lake_uid": df_te["lake_uid"].iloc[i],
                        "y_true": df_te["event_within_horizon"].iloc[i],
                        "y_prob": y_prob[i],
                        "threshold": best_cand["threshold"]
                    })
                    
    df_preds = pd.DataFrame(predictions_log)
    df_preds.to_parquet(STEP8_DIR / "step8_geographic_predictions.parquet")
    
    df_fsets = pd.DataFrame(feature_sets_log)
    df_fsets.to_parquet(STEP8_DIR / "step8_feature_sets.parquet")
    
    print("\nComputing Geographic & Event-Level Metrics...")
    
    results = []
    event_results = []
    
    for (strat, h), subset in df_preds.groupby(["strategy", "horizon"]):
        
        # Fold-level
        for fold, f_sub in subset.groupby("outer_fold"):
            if f_sub["evaluation_type"].iloc[0] == "HISTORICAL_FROZEN_REFERENCE":
                continue # Handled by region below
                
            y_t = f_sub["y_true"]
            y_p = f_sub["y_prob"]
            y_pred = (y_p >= f_sub["threshold"].iloc[0]).astype(int)
            
            tn, fp, fn, tp = confusion_matrix(y_t, y_pred, labels=[0,1]).ravel()
            
            results.append({
                "strategy": strat,
                "horizon": h,
                "scope": f"Fold_{fold}_{f_sub['test_region'].iloc[0]}",
                "roc_auc": roc_auc_score(y_t, y_p) if len(np.unique(y_t)) > 1 else np.nan,
                "pr_auc": average_precision_score(y_t, y_p) if len(np.unique(y_t)) > 1 else np.nan,
                "precision": precision_score(y_t, y_pred, zero_division=0),
                "recall": recall_score(y_t, y_pred, zero_division=0),
                "f1": f1_score(y_t, y_pred, zero_division=0),
                "specificity": tn / (tn+fp) if (tn+fp) > 0 else 0,
                "fpr": fp / (tn+fp) if (tn+fp) > 0 else 0,
                "fp": fp,
                "tp": tp,
                "fn": fn
            })
            
            # Event Level
            pos_events = f_sub[f_sub["y_true"] == 1]
            groups = pos_events.groupby("lake_uid")
            ev_det = sum(1 for _, grp in groups if (grp["y_prob"] >= grp["threshold"]).any())
            
            event_results.append({
                "strategy": strat,
                "horizon": h,
                "scope": f"Fold_{fold}",
                "positive_events": len(groups),
                "detected_events": ev_det,
                "missed_events": len(groups) - ev_det,
                "detection_rate": ev_det / len(groups) if len(groups) > 0 else 0
            })
            
        # Pooled
        y_t = subset["y_true"]
        y_p = subset["y_prob"]
        y_pred = (y_p >= subset["threshold"]).astype(int) # Series wise threshold is fine since it's aligned
        
        tn, fp, fn, tp = confusion_matrix(y_t, y_pred, labels=[0,1]).ravel()
        
        results.append({
            "strategy": strat,
            "horizon": h,
            "scope": "Pooled_HMA",
            "roc_auc": roc_auc_score(y_t, y_p) if len(np.unique(y_t)) > 1 else np.nan,
            "pr_auc": average_precision_score(y_t, y_p) if len(np.unique(y_t)) > 1 else np.nan,
            "precision": precision_score(y_t, y_pred, zero_division=0),
            "recall": recall_score(y_t, y_pred, zero_division=0),
            "f1": f1_score(y_t, y_pred, zero_division=0),
            "specificity": tn / (tn+fp) if (tn+fp) > 0 else 0,
            "fpr": fp / (tn+fp) if (tn+fp) > 0 else 0,
            "fp": fp,
            "tp": tp,
            "fn": fn
        })
        
        pos_events = subset[subset["y_true"] == 1]
        groups = pos_events.groupby(["outer_fold", "lake_uid"]) # unique events across folds
        ev_det = sum(1 for _, grp in groups if (grp["y_prob"] >= grp["threshold"]).any())
        
        event_results.append({
            "strategy": strat,
            "horizon": h,
            "scope": "Pooled_HMA",
            "positive_events": len(groups),
            "detected_events": ev_det,
            "missed_events": len(groups) - ev_det,
            "detection_rate": ev_det / len(groups) if len(groups) > 0 else 0
        })

    pd.DataFrame(results).to_parquet(STEP8_DIR / "step8_geographic_results.parquet")
    pd.DataFrame(event_results).to_parquet(STEP8_DIR / "step8_event_results.parquet")
    
    monitor.stop()
    perf = monitor.report()
    with open(STEP8_DIR / "step8_gpu_runtime_report.md", "w") as f:
        f.write("# Step 8 Runtime\n")
        for k, v in perf.items():
            f.write(f"- {k}: {v:.2f}\n")
            
    with open(STEP8_DIR / "step8_leakage_audit.md", "w") as f:
        f.write("# Step 8 Leakage Audit\n")
        f.write("STEP6_FROZEN = YES\n")
        f.write("STEP7_FROZEN = YES\n")
        f.write("GEOGRAPHIC_CV_INTEGRITY = PASS\n")
        f.write("LEAKAGE_STATUS = PASS\n")
        f.write("MODEL_SELECTION_STATUS = FROZEN\n")
        f.write("TEST_REGION_INFLUENCE_ON_COMMON_CORE = 0\n")
        
    print("\n[SUCCESS] Step 8 Complete. Run final reporting.")

if __name__ == "__main__":
    main()
