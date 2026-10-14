# File: src/metrics.py — metrik evaluasi untuk kejadian langka
import numpy as np
from sklearn.metrics import (roc_auc_score, average_precision_score, precision_score,
                             recall_score, brier_score_loss, f1_score)

def best_threshold(p, y):
    # Cari ambang dengan F1 terbaik. PENTING: panggil ini pada data VALIDASI, bukan uji.
    ths = np.linspace(0.05, 0.95, 19)
    return ths[np.argmax([f1_score(y, p >= t, zero_division=0) for t in ths])]

def report(p, y, p_cal=None, th=0.5):
    r = {'base_rate': y.mean(),                       # PR-AUC model acak = base rate
         'roc_auc': roc_auc_score(y, p),
         'pr_auc': average_precision_score(y, p),
         'precision': precision_score(y, p >= th, zero_division=0),
         'recall': recall_score(y, p >= th, zero_division=0),
         'brier_raw': brier_score_loss(y, p)}
    if p_cal is not None:
        r['brier_cal'] = brier_score_loss(y, p_cal)   # Brier setelah kalibrasi
    return {k: round(float(v), 4) for k, v in r.items()}
