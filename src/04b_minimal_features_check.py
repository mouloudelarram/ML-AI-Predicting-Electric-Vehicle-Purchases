"""
04b_minimal_features_check.py
Teste un set de features MINIMAL (uniquement les interactions les plus fortes identifiées
dans l'audit) contre 'base', pour vérifier si un ajout ciblé (plutôt que kitchen-sink) aide.
Même config rapide / sous-échantillon que 04_feature_engineering_ablation.py.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import NUM_COLS, CAT_COLS, TARGET_COL, SEED
from src.data import load_raw
from src.features import add_engineered_features, add_rank_features
from src.prep import build_matrices
from src.cv_utils import get_folds
from src.models import run_cv, catboost_factory, lightgbm_factory
from src.exp_log import log_experiment

N_SPLITS = 2
SUBSAMPLE_N = 150000

train, test = load_raw()
y_full = train[TARGET_COL].map({"Yes": 1, "No": 0}).astype(int)
train_fe_full = add_engineered_features(train)
test_fe = add_engineered_features(test)

idx_sub, _ = train_test_split(np.arange(len(train_fe_full)), train_size=SUBSAMPLE_N, stratify=y_full, random_state=SEED)
train_fe = train_fe_full.iloc[idx_sub].reset_index(drop=True)
y = y_full.iloc[idx_sub].reset_index(drop=True)

MINIMAL_SETS = {
    "minimal_interactions": [
        "subsidy_x_range_anxiety", "env_x_range_anxiety", "home_x_range_anxiety",
        "ev_readiness_score", "log_income",
    ],
    "minimal_interactions_v2": [
        "subsidy_x_env", "env_x_subsidy_x_home", "subsidy_x_home", "ev_readiness_score",
    ],
    "charging_only": [
        "Total_Charging_Stations", "home_work_station_diff", "charging_density_relative_to_commute",
    ],
    "income_only": ["log_income", "income_per_car", "income_per_commute"],
}

FAST_CB_PARAMS = dict(iterations=600, depth=6, learning_rate=0.09, l2_leaf_reg=3.0,
                       random_strength=1.0, bagging_temperature=1.0, border_count=64,
                       max_ctr_complexity=2)
FAST_LGB_PARAMS = dict(n_estimators=800, learning_rate=0.08, num_leaves=63, max_depth=-1,
                        min_child_samples=30, subsample=0.8, colsample_bytree=0.8, verbose=-1)

results = []
for set_name, extra_cols in MINIMAL_SETS.items():
    cols = NUM_COLS + CAT_COLS + extra_cols
    print(f"\n=== {set_name} ({len(cols)} features) [CatBoost fast] ===")
    Xc, Xc_test, cat_c = build_matrices(train_fe, test_fe, cols, CAT_COLS, "catboost")
    res = run_cv(catboost_factory(FAST_CB_PARAMS, cat_c, seed=SEED, early_stopping_rounds=40), Xc, y, Xc_test, n_splits=N_SPLITS, seed=SEED)
    log_experiment(f"minicheck_catboost_{set_name}", "CatBoost-fast", len(cols), CAT_COLS, FAST_CB_PARAMS, res["scores"], res["oof_auc"], res["fold_scores"], 0, f"minimal FE check {set_name}")
    results.append((set_name, "catboost", res["oof_auc"], res["scores"]["mean"]))

    print(f"=== {set_name} ({len(cols)} features) [LightGBM] ===")
    Xl, Xl_test, cat_l = build_matrices(train_fe, test_fe, cols, CAT_COLS, "native_cat")
    res = run_cv(lightgbm_factory(FAST_LGB_PARAMS, cat_l, seed=SEED, early_stopping_rounds=50), Xl, y, Xl_test, n_splits=N_SPLITS, seed=SEED)
    log_experiment(f"minicheck_lightgbm_{set_name}", "LightGBM", len(cols), CAT_COLS, FAST_LGB_PARAMS, res["scores"], res["oof_auc"], res["fold_scores"], 0, f"minimal FE check {set_name}")
    results.append((set_name, "lightgbm", res["oof_auc"], res["scores"]["mean"]))

print("\n==================== MINIMAL FEATURE CHECK SUMMARY (base ref: cb=0.94014 / lgb=0.94011) ====================")
res_df = pd.DataFrame(results, columns=["feature_set", "model", "oof_auc", "cv_mean"])
print(res_df.to_string(index=False))
res_df.to_csv("outputs/logs/04b_minimal_check_summary.csv", index=False)

