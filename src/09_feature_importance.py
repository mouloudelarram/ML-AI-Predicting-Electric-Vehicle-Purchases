"""
09_feature_importance.py
Importance de features (CatBoost natif + permutation) sur le meilleur modèle final,
et test d'ablation "top 90% / 75% / 50%" des features par importance.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from src.config import NUM_COLS, CAT_COLS, TARGET_COL, SEED, LOG_DIR
from src.data import load_raw
from src.features import add_engineered_features, add_rank_features, ENGINEERED_COLS
from src.prep import build_matrices
from src.cv_utils import get_folds, auc
from src.models import catboost_factory, run_cv

print("Loading + engineering features...")
train, test = load_raw()
y = train[TARGET_COL].map({"Yes": 1, "No": 0}).astype(int)
train_fe = add_engineered_features(train)
test_fe = add_engineered_features(test)
rank_cols = ["Age", "Annual_Income_USD", "Daily_Commute_km"]
train_fe, test_fe = add_rank_features(train_fe, test_fe, rank_cols)
FEATURE_COLS = NUM_COLS + CAT_COLS + ENGINEERED_COLS + [f"{c}_rank" for c in rank_cols]

best_params_path = os.path.join(LOG_DIR, "best_params.json")
params = dict(iterations=2000, depth=6, learning_rate=0.06, l2_leaf_reg=3.0, max_ctr_complexity=2, border_count=64)
if os.path.exists(best_params_path):
    with open(best_params_path) as f:
        bp = json.load(f)
    if "catboost" in bp:
        params.update(bp["catboost"]["params"])
        params["iterations"] = 2000

Xc, Xc_test, cat_c = build_matrices(train_fe, test_fe, FEATURE_COLS, CAT_COLS, "catboost")

# split simple pour importance (un seul fold, rapide)
folds = get_folds(y, n_splits=5, seed=SEED)
tr_idx, val_idx = folds[0]
from catboost import CatBoostClassifier
model = CatBoostClassifier(**params, cat_features=cat_c, eval_metric="AUC", early_stopping_rounds=100, verbose=False)
model.fit(Xc.iloc[tr_idx], y.iloc[tr_idx], eval_set=(Xc.iloc[val_idx], y.iloc[val_idx]), use_best_model=True)

# --- importance native CatBoost ---
imp = model.get_feature_importance(prettified=True)
imp.columns = ["feature", "importance"]
imp = imp.sort_values("importance", ascending=False).reset_index(drop=True)
print("\n--- CatBoost native feature importance ---")
print(imp.to_string(index=False))
imp.to_csv(os.path.join(LOG_DIR, "feature_importance_catboost.csv"), index=False)

# --- permutation importance (sur un sous-échantillon de validation pour la vitesse) ---
val_sub_idx = val_idx[:30000] if len(val_idx) > 30000 else val_idx
print("\nCalcul de la permutation importance (sous-échantillon)...")


class CBWrapper:
    def fit(self, X, y_):
        return self

    def predict_proba(self, X):
        return model.predict_proba(X)

    def score(self, X, y_):
        return auc(y_, model.predict_proba(X)[:, 1])


wrapper = CBWrapper()
perm = permutation_importance(wrapper, Xc.iloc[val_sub_idx], y.iloc[val_sub_idx], n_repeats=5, random_state=SEED, scoring=None)
perm_df = pd.DataFrame({"feature": Xc.columns, "perm_importance_mean": perm.importances_mean, "perm_importance_std": perm.importances_std})
perm_df = perm_df.sort_values("perm_importance_mean", ascending=False).reset_index(drop=True)
print("\n--- Permutation importance (AUC drop) ---")
print(perm_df.to_string(index=False))
perm_df.to_csv(os.path.join(LOG_DIR, "feature_importance_permutation.csv"), index=False)

# --- test top 90% / 75% / 50% des features (par importance CatBoost) ---
ranked_features = imp["feature"].tolist()
n_total = len(ranked_features)
subsets = {
    "top100pct": ranked_features,
    "top90pct": ranked_features[: int(n_total * 0.9)],
    "top75pct": ranked_features[: int(n_total * 0.75)],
    "top50pct": ranked_features[: int(n_total * 0.5)],
}

print("\n--- Test de sélection de features (3-fold CV, CatBoost fast) ---")
sel_results = []
for name, feats in subsets.items():
    cat_subset = [c for c in cat_c if c in feats]
    Xs, Xs_test, _ = build_matrices(train_fe, test_fe, feats, cat_subset, "catboost")
    res = run_cv(catboost_factory(dict(iterations=1200, depth=6, learning_rate=0.08, max_ctr_complexity=2, border_count=64),
                                   cat_subset, seed=SEED, early_stopping_rounds=60),
                 Xs, y, Xs_test, n_splits=3, seed=SEED, verbose=False)
    print(f"  {name:12s} ({len(feats)} feats): OOF AUC={res['oof_auc']:.5f}  mean={res['scores']['mean']:.5f}")
    sel_results.append((name, len(feats), res["oof_auc"], res["scores"]["mean"]))

sel_df = pd.DataFrame(sel_results, columns=["subset", "n_features", "oof_auc", "cv_mean"])
sel_df.to_csv(os.path.join(LOG_DIR, "feature_selection_results.csv"), index=False)
print("\nDone.")

