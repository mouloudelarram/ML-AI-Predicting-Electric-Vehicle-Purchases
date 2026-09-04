"""
02_baselines_stage_a.py
Stage A (rapide) : baselines pour les 7 familles de modèles sur les features BRUTES (sans FE),
validation croisée 3-fold, pour valider le pipeline et obtenir un premier classement.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from src.config import NUM_COLS, CAT_COLS, TARGET_COL, ID_COL, SEED
from src.data import load_raw
from src.prep import build_matrices
from src.models import (
    run_cv, catboost_factory, lightgbm_factory, xgboost_factory,
    histgb_factory, extratrees_factory, randomforest_factory, logreg_factory,
)
from src.exp_log import log_experiment

N_SPLITS = 3
FEATURE_COLS = NUM_COLS + CAT_COLS

print("Loading data...")
train, test = load_raw()
y = train[TARGET_COL].map({"Yes": 1, "No": 0}).astype(int)
print(f"train shape={train.shape}  test shape={test.shape}  positive rate={y.mean():.4f}")

results = {}

# ---------------------------------------------------------------------------
# CatBoost (native categorical)
# ---------------------------------------------------------------------------
print("\n=== CatBoost (base features) ===")
Xc, Xc_test, cat_c = build_matrices(train, test, FEATURE_COLS, CAT_COLS, "catboost")
params = dict(iterations=3000, depth=6, learning_rate=0.06, l2_leaf_reg=3.0, random_strength=1.0,
              bagging_temperature=1.0, border_count=128, grow_policy="SymmetricTree")
t0 = time.time()
res = run_cv(catboost_factory(params, cat_c, seed=SEED), Xc, y, Xc_test, n_splits=N_SPLITS, seed=SEED)
dt = time.time() - t0
np.save("outputs/oof/stageA_catboost_base_oof.npy", res["oof"])
np.save("outputs/preds/stageA_catboost_base_test.npy", res["test_pred"])
log_experiment("stageA_catboost_base", "CatBoost", len(FEATURE_COLS), CAT_COLS, params, res["scores"], res["oof_auc"], res["fold_scores"], dt, "stage A base features 3-fold")
results["catboost"] = res

# ---------------------------------------------------------------------------
# LightGBM (native categorical)
# ---------------------------------------------------------------------------
print("\n=== LightGBM (base features) ===")
Xl, Xl_test, cat_l = build_matrices(train, test, FEATURE_COLS, CAT_COLS, "native_cat")
params = dict(n_estimators=3000, learning_rate=0.05, num_leaves=63, max_depth=-1,
              min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
              reg_alpha=0.0, reg_lambda=0.0, verbose=-1)
t0 = time.time()
res = run_cv(lightgbm_factory(params, cat_l, seed=SEED), Xl, y, Xl_test, n_splits=N_SPLITS, seed=SEED)
dt = time.time() - t0
np.save("outputs/oof/stageA_lightgbm_base_oof.npy", res["oof"])
np.save("outputs/preds/stageA_lightgbm_base_test.npy", res["test_pred"])
log_experiment("stageA_lightgbm_base", "LightGBM", len(FEATURE_COLS), CAT_COLS, params, res["scores"], res["oof_auc"], res["fold_scores"], dt, "stage A base features 3-fold")
results["lightgbm"] = res

# ---------------------------------------------------------------------------
# XGBoost (native categorical)
# ---------------------------------------------------------------------------
print("\n=== XGBoost (base features) ===")
Xx, Xx_test, cat_x = build_matrices(train, test, FEATURE_COLS, CAT_COLS, "native_cat")
params = dict(n_estimators=3000, learning_rate=0.05, max_depth=6, min_child_weight=5,
              subsample=0.8, colsample_bytree=0.8, gamma=0.0, reg_alpha=0.0, reg_lambda=1.0)
t0 = time.time()
res = run_cv(xgboost_factory(params, seed=SEED), Xx, y, Xx_test, n_splits=N_SPLITS, seed=SEED)
dt = time.time() - t0
np.save("outputs/oof/stageA_xgboost_base_oof.npy", res["oof"])
np.save("outputs/preds/stageA_xgboost_base_test.npy", res["test_pred"])
log_experiment("stageA_xgboost_base", "XGBoost", len(FEATURE_COLS), CAT_COLS, params, res["scores"], res["oof_auc"], res["fold_scores"], dt, "stage A base features 3-fold")
results["xgboost"] = res

# ---------------------------------------------------------------------------
# HistGradientBoosting
# ---------------------------------------------------------------------------
print("\n=== HistGradientBoosting (base features) ===")
Xh, Xh_test, cat_h = build_matrices(train, test, FEATURE_COLS, CAT_COLS, "native_cat")
params = dict(max_iter=1000, learning_rate=0.06, max_depth=None, max_leaf_nodes=63,
              min_samples_leaf=50, l2_regularization=0.0, early_stopping=True,
              n_iter_no_change=30, validation_fraction=0.1)
t0 = time.time()
res = run_cv(histgb_factory(params, seed=SEED), Xh, y, Xh_test, n_splits=N_SPLITS, seed=SEED)
dt = time.time() - t0
np.save("outputs/oof/stageA_histgb_base_oof.npy", res["oof"])
np.save("outputs/preds/stageA_histgb_base_test.npy", res["test_pred"])
log_experiment("stageA_histgb_base", "HistGB", len(FEATURE_COLS), CAT_COLS, params, res["scores"], res["oof_auc"], res["fold_scores"], dt, "stage A base features 3-fold")
results["histgb"] = res

# ---------------------------------------------------------------------------
# One-hot based matrix (reused for ExtraTrees / RandomForest / LogReg)
# ---------------------------------------------------------------------------
print("\nBuilding one-hot matrices for ExtraTrees/RandomForest/LogReg...")
Xo, Xo_test, _ = build_matrices(train, test, FEATURE_COLS, CAT_COLS, "ohe")
print(f"OHE matrix shape: {Xo.shape}")

print("\n=== ExtraTrees (base features, OHE) ===")
params = dict(n_estimators=400, max_depth=18, min_samples_leaf=5, max_features="sqrt")
t0 = time.time()
res = run_cv(extratrees_factory(params, seed=SEED), Xo, y, Xo_test, n_splits=N_SPLITS, seed=SEED)
dt = time.time() - t0
np.save("outputs/oof/stageA_extratrees_base_oof.npy", res["oof"])
np.save("outputs/preds/stageA_extratrees_base_test.npy", res["test_pred"])
log_experiment("stageA_extratrees_base", "ExtraTrees", Xo.shape[1], [], params, res["scores"], res["oof_auc"], res["fold_scores"], dt, "stage A base features 3-fold, OHE")
results["extratrees"] = res

print("\n=== RandomForest (base features, OHE) ===")
params = dict(n_estimators=400, max_depth=18, min_samples_leaf=5, max_features="sqrt")
t0 = time.time()
res = run_cv(randomforest_factory(params, seed=SEED), Xo, y, Xo_test, n_splits=N_SPLITS, seed=SEED)
dt = time.time() - t0
np.save("outputs/oof/stageA_randomforest_base_oof.npy", res["oof"])
np.save("outputs/preds/stageA_randomforest_base_test.npy", res["test_pred"])
log_experiment("stageA_randomforest_base", "RandomForest", Xo.shape[1], [], params, res["scores"], res["oof_auc"], res["fold_scores"], dt, "stage A base features 3-fold, OHE")
results["randomforest"] = res

print("\n=== LogisticRegression (base features, OHE + scaled) ===")
params = dict(max_iter=500, C=1.0, solver="lbfgs")
t0 = time.time()
res = run_cv(logreg_factory(params, seed=SEED), Xo, y, Xo_test, n_splits=N_SPLITS, seed=SEED)
dt = time.time() - t0
np.save("outputs/oof/stageA_logreg_base_oof.npy", res["oof"])
np.save("outputs/preds/stageA_logreg_base_test.npy", res["test_pred"])
log_experiment("stageA_logreg_base", "LogReg", Xo.shape[1], [], params, res["scores"], res["oof_auc"], res["fold_scores"], dt, "stage A base features 3-fold, OHE+scaled")
results["logreg"] = res

print("\n\n==================== SUMMARY (Stage A, base features, 3-fold) ====================")
for name, res in results.items():
    print(f"{name:15s}  OOF AUC={res['oof_auc']:.5f}  mean={res['scores']['mean']:.5f}  std={res['scores']['std']:.5f}  time={res['total_time']:.1f}s")

