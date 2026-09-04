"""Fonctions de blending: moyenne pondérée de probabilités, moyenne de rangs, stacking."""
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from src.cv_utils import get_folds


def optimize_weights_random_search(pred_dict, y, n_iter=2500, seed=42, refine_rounds=3, search_subsample=150000):
    """Recherche aléatoire (Dirichlet) sur le simplexe des poids + raffinement local.

    Pour accélérer sur de gros datasets, la recherche est faite sur un sous-échantillon
    stratifié ; le score final rapporté est toujours recalculé sur l'ensemble COMPLET.
    """
    rng = np.random.default_rng(seed)
    names = list(pred_dict.keys())
    P = np.column_stack([pred_dict[n] for n in names])
    y_arr = np.asarray(y)
    n = len(names)

    if search_subsample is not None and search_subsample < len(y_arr):
        pos_idx = np.where(y_arr == 1)[0]
        neg_idx = np.where(y_arr == 0)[0]
        frac = search_subsample / len(y_arr)
        n_pos = max(1, int(len(pos_idx) * frac))
        n_neg = max(1, int(len(neg_idx) * frac))
        sub_idx = np.concatenate([
            rng.choice(pos_idx, size=n_pos, replace=False),
            rng.choice(neg_idx, size=n_neg, replace=False),
        ])
        P_search, y_search = P[sub_idx], y_arr[sub_idx]
    else:
        P_search, y_search = P, y_arr

    def score_of(w, PP, yy):
        return roc_auc_score(yy, PP @ w)

    best_w = np.ones(n) / n
    best_score = score_of(best_w, P_search, y_search)

    alphas = np.ones(n)
    for _ in range(n_iter):
        w = rng.dirichlet(alphas)
        s = score_of(w, P_search, y_search)
        if s > best_score:
            best_score, best_w = s, w

    # raffinement local: perturbations gaussiennes décroissantes autour du meilleur point
    for r in range(refine_rounds):
        scale = 0.2 / (r + 1)
        for _ in range(max(50, n_iter // 4)):
            w = best_w + rng.normal(0, scale, size=n)
            w = np.clip(w, 0, None)
            if w.sum() <= 0:
                continue
            w = w / w.sum()
            s = score_of(w, P_search, y_search)
            if s > best_score:
                best_score, best_w = s, w

    # score final honnête sur l'ensemble COMPLET
    final_score = score_of(best_w, P, y_arr)
    return dict(zip(names, best_w)), final_score


def rank_transform(pred_dict):
    """Transforme chaque vecteur de prédictions en rang percentile (0-1), indépendamment."""
    return {k: rankdata(v) / len(v) for k, v in pred_dict.items()}


def blend_predictions(pred_dict, weights):
    names = list(pred_dict.keys())
    P = np.column_stack([pred_dict[n] for n in names])
    w = np.array([weights[n] for n in names])
    return P @ w


def stacking_cv(oof_dict, y, n_splits=5, seed=42, C=1.0):
    """Stacking logistic-regression, validé en CV pour une estimation honnête de l'AUC meta."""
    names = list(oof_dict.keys())
    X_meta = np.column_stack([oof_dict[n] for n in names])
    y_arr = np.asarray(y)
    folds = get_folds(y_arr, n_splits=n_splits, seed=seed)
    meta_oof = np.zeros(len(y_arr))
    for tr_idx, val_idx in folds:
        m = LogisticRegression(C=C, max_iter=1000)
        m.fit(X_meta[tr_idx], y_arr[tr_idx])
        meta_oof[val_idx] = m.predict_proba(X_meta[val_idx])[:, 1]
    meta_auc = roc_auc_score(y_arr, meta_oof)

    # modèle final entraîné sur 100% des OOF (pour appliquer aux prédictions test)
    final_meta = LogisticRegression(C=C, max_iter=1000)
    final_meta.fit(X_meta, y_arr)
    return meta_oof, meta_auc, final_meta, names


def apply_stacking_to_test(final_meta, test_pred_dict, names):
    X_meta_test = np.column_stack([test_pred_dict[n] for n in names])
    return final_meta.predict_proba(X_meta_test)[:, 1]


