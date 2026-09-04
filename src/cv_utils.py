"""Utilitaires de validation croisée."""
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score


def get_folds(y, n_splits=5, seed=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    y_arr = np.asarray(y)
    return list(skf.split(np.zeros(len(y_arr)), y_arr))


def summarize_scores(fold_scores):
    arr = np.array(fold_scores)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def auc(y_true, y_pred):
    return roc_auc_score(y_true, y_pred)

