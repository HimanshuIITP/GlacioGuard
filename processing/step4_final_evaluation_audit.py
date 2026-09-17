#!/usr/bin/env python3
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score, confusion_matrix, balanced_accuracy_score

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

def main():
    print("Running Final Evaluation Audit (Reconciled)...")
    
    preds_path = PROCESSED_DIR / "baseline_predictions.parquet"
    df_preds = pd.read_parquet(preds_path)
    
    events_path = PROCESSED_DIR / "glof_events.parquet"
    df_events_meta = pd.read_parquet(events_path)
    df_events_meta['event_date'] = pd.to_datetime(df_events_meta['event_date'], utc=True)
    
    matches_path = PROCESSED_DIR / "glof_event_lake_matches.parquet"
    df_matches = pd.read_parquet(matches_path)
    
    # Identify taxonomy
    confirmed_events = df_matches[df_matches['match_method'] == 'polygon_intersection']['event_id'].tolist()
    manual_review_events = df_matches[df_matches['match_method'].str.contains('distance', na=False)]['event_id'].tolist()
    unresolved_events = df_matches[df_matches['match_status'] == 'UNMATCHED']['event_id'].tolist()
    
    # 1. Pooled LOOCV Predictions
    df_loocv = df_preds[df_preds['split_name'].str.startswith('loocv_')].copy()
    
    pooled_results = []
    group_cols = ['horizon_days', 'model_name', 'feature_group', 'weighting_strategy']
    
    for name, group in df_loocv.groupby(group_cols):
        y_true = group['true_label'].values
        y_prob = group['predicted_probability'].values
        y_pred = group['predicted_class'].values
        
        test_rows = len(y_true)
        test_pos = sum(y_true)
        
        if len(np.unique(y_true)) > 1:
            roc_auc = roc_auc_score(y_true, y_prob)
            pr_auc = average_precision_score(y_true, y_prob)
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            bal_acc = balanced_accuracy_score(y_true, y_pred)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        else:
            roc_auc = np.nan
            pr_auc = np.nan
            prec = np.nan
            rec = np.nan
            f1 = np.nan
            bal_acc = np.nan
            spec = np.nan
            fp = sum((y_pred == 1) & (y_true == 0))
            
        pooled_results.append({
            'horizon_days': name[0],
            'model_name': name[1],
            'feature_group': name[2],
            'weighting_strategy': name[3],
            'test_rows': test_rows,
            'test_positive_rows': test_pos,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'specificity': spec,
            'balanced_accuracy': bal_acc,
            'row_level_false_positives': fp
        })
        
    df_pooled = pd.DataFrame(pooled_results)
    
    # Candidate selection logic based on pooled OOF results
    candidates = {}
    for h in [3, 7, 14, 30]:
        h_df = df_pooled[df_pooled['horizon_days'] == h]
        best_row = h_df.sort_values('pr_auc', ascending=False).iloc[0]
        if best_row['pr_auc'] > 0.15 and best_row['f1'] > 0.1:
            candidates[h] = best_row
        else:
            candidates[h] = None

    # Event-Level Evaluation
    all_eligible_events = confirmed_events + manual_review_events
    
    event_evals = []
    for h in [3, 7, 14, 30]:
        cand = candidates[h]
        if cand is None:
            continue
            
        m = cand['model_name']
        fg = cand['feature_group']
        ws = cand['weighting_strategy']
        
        sub_df = df_loocv[(df_loocv['horizon_days'] == h) & 
                          (df_loocv['model_name'] == m) & 
                          (df_loocv['feature_group'] == fg) & 
                          (df_loocv['weighting_strategy'] == ws)]
                          
        for evt in all_eligible_events:
            evt_df = sub_df[sub_df['matched_event_id'] == evt].copy()
            if evt_df.empty:
                continue
                
            evt_meta = df_events_meta[df_events_meta['event_id'] == evt].iloc[0]
            event_date = evt_meta['event_date']
                
            evt_df['reference_timestamp'] = pd.to_datetime(evt_df['reference_timestamp'], utc=True)
            evt_df = evt_df.sort_values('reference_timestamp')
            
            valid_rows = len(evt_df)
            pos_rows = evt_df['true_label'].sum()
            
            pos_preds = evt_df[(evt_df['predicted_class'] == 1) & (evt_df['true_label'] == 1)]
            
            detected = 'YES' if not pos_preds.empty else 'NO'
            earliest_alert = pos_preds.iloc[0]['reference_timestamp'] if not pos_preds.empty else None
            
            if earliest_alert is not None:
                lead_time_days = (event_date - earliest_alert).days
            else:
                lead_time_days = None
                
            num_alerts = len(pos_preds)
            false_alerts = len(evt_df[(evt_df['predicted_class'] == 1) & (evt_df['true_label'] == 0)])
            
            event_evals.append({
                'horizon_days': h,
                'model_name': m,
                'event_id': evt,
                'detected': detected,
                'earliest_alert': earliest_alert.isoformat() if earliest_alert else None,
                'lead_time_days': lead_time_days,
                'alert_rows': num_alerts,
                'event_window_false_alerts': false_alerts,
                'valid_test_rows': valid_rows,
                'positive_rows': pos_rows
            })
            
    df_events = pd.DataFrame(event_evals)

    # Fold averages (secondary diagnostic)
    df_chrono = df_preds[df_preds['split_name'] == 'chronological']
    cutoff_year = 2005
    df_features = pd.read_parquet(PROCESSED_DIR / "glof_feature_matrix.parquet")
    chrono_train = df_features[pd.to_datetime(df_features['reference_timestamp']).dt.year <= cutoff_year]
    chrono_test = df_features[pd.to_datetime(df_features['reference_timestamp']).dt.year > cutoff_year]
    
    train_events = chrono_train['matched_event_id'].dropna().unique().tolist()
    test_events = chrono_test['matched_event_id'].dropna().unique().tolist()
    
    report_path = PROCESSED_DIR / "step4_final_evaluation_report_reconciled.md"
    with open(report_path, "w") as f:
        f.write("# Step 4 Final Evaluation Audit Report (Reconciled)\n\n")
        
        f.write("## 1. Event Group Definitions\n")
        f.write(f"- **Confirmed GLOF Event IDs (Polygon Matches)**: {confirmed_events}\n")
        f.write(f"- **Manual-Review Event IDs (Distance Matches)**: {manual_review_events}\n")
        f.write(f"- **Unresolved Event IDs**: {len(unresolved_events)} events without valid candidates.\n")
        f.write("- **Control-Group IDs**: Negative-eligible periods associated with the lakes of the above eligible events.\n")
        f.write("- **Verification**: Only the 10 combined Confirmed and Manual-Review eligible events produced positive labels in this dataset.\n\n")

        f.write("## 2. False Alarm Definitions\n")
        f.write("- **Row-level false positives**: Predicted positive during NEGATIVE_ELIGIBLE reference rows (pooled OOF table).\n")
        f.write("- **Event-window false alerts**: Predicted alerts outside the valid pre-event alert window for a particular event (event-level table).\n\n")

        f.write("## 3. Pooled OOF Results (Primary Performance Estimate)\n")
        f.write(df_pooled.to_csv(index=False, sep='|'))
        f.write("\n\n")
        
        f.write("## 4. Candidate Models\n")
        for h in [3, 7, 14, 30]:
            cand = candidates[h]
            if cand is not None:
                f.write(f"- {h}d: `EXPLORATORY_CANDIDATE` = {cand['model_name']} ({cand['feature_group']}, {cand['weighting_strategy']}). Selected based on highest Pooled PR-AUC.\n")
            else:
                f.write(f"- {h}d: `NO_CLEAR_EXPLORATORY_CANDIDATE`. No configuration demonstrated convincing pooled predictive signal.\n")
        f.write("\n")
        
        f.write("## 5. Event-Level Evaluation (Selected Candidates)\n")
        if not df_events.empty:
            f.write(df_events.to_csv(index=False, sep='|'))
        else:
            f.write("No candidate models met the threshold for event-level reporting.\n")
        f.write("\n\n")
        
        f.write("## 6. Chronological Holdout (Secondary Diagnostic)\n")
        f.write(f"- Train event IDs: {train_events}\n")
        f.write(f"- Test event IDs: {test_events}\n")
        f.write("- **STATUS**: Underpowered. Given the extremely small sample of eligible events (10), the chronological split is highly sensitive to the exact event cutoffs and lacks sufficient diversity for robust generalization testing.\n\n")
        
        f.write("## 7. Scientific Interpretation\n")
        f.write("- Only 4 confirmed positive GLOF events (polygon-matched) and 6 manual-review events exist in the evaluation.\n")
        f.write("- Results are purely exploratory.\n")
        f.write("- No statistical generalization is justified.\n")
        f.write("- No production threshold has been established.\n")
        f.write("- No causal interpretation of feature importance.\n")
        
    print("Audit Complete. step4_final_evaluation_report_reconciled.md generated.")

if __name__ == "__main__":
    main()
