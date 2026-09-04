"""
04_feature_engineering_ablation.py
Compare plusieurs familles de features (base / +engineered / +rank / +target-encoding)
avec CatBoost (config rapide) et LightGBM, sur un SOUS-ÉCHANTILLON stratifié + CV à 2 folds,
pour décider rapidement quelles features garder (Stage A - itération rapide).
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import NUM_COLS, CAT_COLS, TARGET_COL, SEED
from src.data import load_raw
from src.features import add_engineered_features, add_rank_features, target_encode_oof, ENGINEERED_COLS
from src.prep import build_matrices
from src.cv_utils import get_folds
from src.models import run_cv, catboost_factory, lightgbm_factory
from src.exp_log import log_experiment

N_SPLITS = 2
SUBSAMPLE_N = 150000

print("Loading data...")
train, test = load_raw()
y_full = train[TARGET_COL].map({"Yes": 1, "No": 0}).astype(int)

print("Engineering features...")
train_fe_full = add_engineered_features(train)
test_fe = add_engineered_features(test)

rank_cols = ["Age", "Annual_Income_USD", "Daily_Commute_km"]
train_fe_full, test_fe = add_rank_features(train_fe_full, test_fe, rank_cols)
rank_feat_names = [f"{c}_rank" for c in rank_cols]

# --- sous-échantillon stratifié pour accélérer l'ablation ---
idx_sub, _ = train_test_split(np.arange(len(train_fe_full)), train_size=SUBSAMPLE_N, stratify=y_full, random_state=SEED)
train_fe = train_fe_full.iloc[idx_sub].reset_index(drop=True)
y = y_full.iloc[idx_sub].reset_index(drop=True)
print(f"Ablation subsample: {len(train_fe)} rows (full train={len(train_fe_full)})")

folds = get_folds(y, n_splits=N_SPLITS, seed=SEED)

te_cols = [
    "City_Type", "Current_Car_Type",
    ("City_Type", "Home_Charging_Possible"),
    ("Current_Car_Type", "Range_Anxiety_Level"),
    ("Subsidy_Available", "Range_Anxiety_Level"),
]
print("Target-encoding OOF (pour tests LightGBM)...")
train_fe_num_target = train_fe.assign(**{TARGET_COL: y.to_numpy()})
train_fe, test_fe_te = target_encode_oof(
    train_fe_num_target, test_fe, te_cols, TARGET_COL, folds, smoothing=30.0,
)
te_feat_names = [f"te_{c if isinstance(c, str) else '_'.join(c)}" for c in te_cols]

FEATURE_SETS = {
    "base": NUM_COLS + CAT_COLS,
    "base_plus_engineered": NUM_COLS + CAT_COLS + ENGINEERED_COLS,
    "base_plus_engineered_plus_rank": NUM_COLS + CAT_COLS + ENGINEERED_COLS + rank_feat_names,
    "base_plus_engineered_plus_te": NUM_COLS + CAT_COLS + ENGINEERED_COLS + rank_feat_names + te_feat_names,
}

# Config CatBoost TRÈS rapide pour l'ablation (sous-échantillon, peu d'itérations)
FAST_CB_PARAMS = dict(iterations=600, depth=6, learning_rate=0.09, l2_leaf_reg=3.0,
                       random_strength=1.0, bagging_temperature=1.0, border_count=64,
                       max_ctr_complexity=2)
FAST_LGB_PARAMS = dict(n_estimators=800, learning_rate=0.08, num_leaves=63, max_depth=-1,
                        min_child_samples=30, subsample=0.8, colsample_bytree=0.8, verbose=-1)

results = []
for set_name, cols in FEATURE_SETS.items():
    print(f"\n=== Feature set: {set_name} ({len(cols)} features) [CatBoost fast] ===")
    Xc, Xc_test, cat_c = build_matrices(train_fe, test_fe_te, cols, CAT_COLS, "catboost")
    t0 = time.time()
    res = run_cv(catboost_factory(FAST_CB_PARAMS, cat_c, seed=SEED, early_stopping_rounds=40), Xc, y, Xc_test, n_splits=N_SPLITS, seed=SEED)
    dt = time.time() - t0
    exp_id = f"ablation_catboost_{set_name}"
    np.save(f"outputs/oof/{exp_id}_oof.npy", res["oof"])
    np.save(f"outputs/preds/{exp_id}_test.npy", res["test_pred"])
    log_experiment(exp_id, "CatBoost-fast", len(cols), CAT_COLS, FAST_CB_PARAMS, res["scores"], res["oof_auc"], res["fold_scores"], dt, f"ablation FE set={set_name} (subsample {SUBSAMPLE_N})")
    results.append((set_name, "catboost", res["oof_auc"], res["scores"]["mean"], res["scores"]["std"]))

    print(f"\n=== Feature set: {set_name} ({len(cols)} features) [LightGBM] ===")
    Xl, Xl_test, cat_l = build_matrices(train_fe, test_fe_te, cols, CAT_COLS, "native_cat")
    t0 = time.time()
    res = run_cv(lightgbm_factory(FAST_LGB_PARAMS, cat_l, seed=SEED, early_stopping_rounds=50), Xl, y, Xl_test, n_splits=N_SPLITS, seed=SEED)
    dt = time.time() - t0
    exp_id = f"ablation_lightgbm_{set_name}"
    np.save(f"outputs/oof/{exp_id}_oof.npy", res["oof"])
    np.save(f"outputs/preds/{exp_id}_test.npy", res["test_pred"])
    log_experiment(exp_id, "LightGBM", len(cols), CAT_COLS, FAST_LGB_PARAMS, res["scores"], res["oof_auc"], res["fold_scores"], dt, f"ablation FE set={set_name} (subsample {SUBSAMPLE_N})")
    results.append((set_name, "lightgbm", res["oof_auc"], res["scores"]["mean"], res["scores"]["std"]))

print("\n\n==================== ABLATION SUMMARY ====================")
res_df = pd.DataFrame(results, columns=["feature_set", "model", "oof_auc", "cv_mean", "cv_std"])
print(res_df.to_string(index=False))
res_df.to_csv("outputs/logs/04_ablation_summary.csv", index=False)




