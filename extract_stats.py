import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score, average_precision_score

df = pd.read_parquet('data/processed/step6/step6_outer_predictions.parquet')
for h in [3, 7, 14, 30]:
    print(f'=== HORIZON {h} ===')
    best_pr = -1
    best_stats = {}
    for (fg, smote), sub in df[df['horizon'] == h].groupby(['feature_group', 'smote']):
        y_t = sub['y_true']
        y_p = sub['y_prob']
        try:
            pr = average_precision_score(y_t, y_p)
            if pr > best_pr:
                best_pr = pr
                best_stats = {'fg': fg, 'smote': smote, 'df': sub, 'pr': pr, 'roc': roc_auc_score(y_t, y_p)}
        except: pass
    if best_stats:
        sub = best_stats['df']
        y_t = sub['y_true']
        y_p = sub['y_prob']
        y_pred = (y_p >= sub['threshold']).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_t, y_pred, labels=[0,1]).ravel()
        print(f"Feature Group: {best_stats['fg']} | SMOTE: {best_stats['smote']}")
        print(f"PR-AUC: {best_stats['pr']:.4f}")
        print(f"ROC-AUC: {best_stats['roc']:.4f}")
        print(f"F1: {f1_score(y_t, y_pred):.4f}")
        print(f"Recall: {recall_score(y_t, y_pred):.4f}")
        print(f"False Positives: {fp}")
        
        pos_events = sub[sub['y_true'] == 1]
        groups = pos_events.groupby('cv_group_id')
        ev_det = sum(1 for _, grp in groups if (grp['y_prob'] >= grp['threshold']).any())
        ev_total = len(groups)
        print(f"Event-level detection: {ev_det}/{ev_total} ({(ev_det/ev_total if ev_total > 0 else 0):.2f})\n")
