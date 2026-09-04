"""
06_stage_b_final_models.py
Validation finale 5-fold CV sur les modèles candidats (hyperparamètres issus de la recherche
Optuna si disponibles). Sauvegarde OOF + test predictions "final_<model>".

Choix du feature set (voir 04_feature_engineering_ablation.py + 04b_minimal_features_check.py):
16/16 comparaisons (CatBoost+LightGBM, 8 variantes de FE) ont montré que les features BRUTES
surpassent toutes les variantes enrichies -> les modèles arbres natifs utilisent le set de base.
La Logistic Regression (sanity baseline, modèle linéaire) utilise elle le set enrichi, car un
modèle linéaire ne peut pas découvrir seul les interactions/ratios.
"""
import sys, os, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from src.config import NUM_COLS, CAT_COLS, TARGET_COL, SEED, LOG_DIR, OOF_DIR, PRED_DIR
from src.data import load_raw
from src.features import add_engineered_features, add_rank_features, ENGINEERED_COLS
from src.prep import build_matrices
from src.models import (
    run_cv, catboost_factory, lightgbm_factory, xgboost_factory,
    histgb_factory, extratrees_factory, randomforest_factory, logreg_factory,
)
from src.exp_log import log_experiment

parser = argparse.ArgumentParser()
parser.add_argument("--n_splits", type=int, default=5)
parser.add_argument("--models", type=str, default="catboost,lightgbm,xgboost,extratrees,histgb,randomforest,logreg")
args = parser.parse_args()
N_SPLITS = args.n_splits

print("Loading + engineering features...")
train, test = load_raw()
y = train[TARGET_COL].map({"Yes": 1, "No": 0}).astype(int)
train_fe = add_engineered_features(train)
test_fe = add_engineered_features(test)
rank_cols = ["Age", "Annual_Income_USD", "Daily_Commute_km"]
train_fe, test_fe = add_rank_features(train_fe, test_fe, rank_cols)
RANK_FEATS = [f"{c}_rank" for c in rank_cols]

BASE_FEATURE_COLS = NUM_COLS + CAT_COLS
ENGINEERED_FEATURE_COLS = NUM_COLS + CAT_COLS + ENGINEERED_COLS + RANK_FEATS
print(f"Base feature set (arbres natifs): {len(BASE_FEATURE_COLS)} features")
print(f"Enriched feature set (LogReg)   : {len(ENGINEERED_FEATURE_COLS)} features")

best_params_path = os.path.join(LOG_DIR, "best_params.json")
best_params = {}
if os.path.exists(best_params_path):
    with open(best_params_path) as f:
        best_params = json.load(f)
    print(f"Loaded tuned hyperparameters for: {list(best_params.keys())}")

models_to_run = args.models.split(",")
summary = []

# ---------------------------------------------------------------------------
# CatBoost
# ---------------------------------------------------------------------------
if "catboost" in models_to_run:
    print(f"\n=== [FINAL] CatBoost  ({N_SPLITS}-fold) ===")
    p = dict(best_params.get("catboost", {}).get("params", dict(
        depth=6, learning_rate=0.05, l2_leaf_reg=3.0, random_strength=1.0,
        bagging_temperature=1.0, border_count=128)))
    p["iterations"] = 6000
    p.setdefault("max_ctr_complexity", 2)
    Xc, Xc_test, cat_c = build_matrices(train_fe, test_fe, BASE_FEATURE_COLS, CAT_COLS, "catboost")
    t0 = time.time()
    res = run_cv(catboost_factory(p, cat_c, seed=SEED, early_stopping_rounds=150), Xc, y, Xc_test, n_splits=N_SPLITS, seed=SEED)
    dt = time.time() - t0
    np.save(os.path.join(OOF_DIR, "final_catboost_oof.npy"), res["oof"])
    np.save(os.path.join(PRED_DIR, "final_catboost_test.npy"), res["test_pred"])
    log_experiment("final_catboost", "CatBoost", len(BASE_FEATURE_COLS), CAT_COLS, p, res["scores"], res["oof_auc"], res["fold_scores"], dt, f"STAGE B final, {N_SPLITS}-fold, tuned params, base features")
    summary.append(("catboost", res["oof_auc"], res["scores"]["mean"], res["scores"]["std"]))

# ---------------------------------------------------------------------------
# LightGBM
# ---------------------------------------------------------------------------
if "lightgbm" in models_to_run:
    print(f"\n=== [FINAL] LightGBM  ({N_SPLITS}-fold) ===")
    p = dict(best_params.get("lightgbm", {}).get("params", dict(
        num_leaves=63, max_depth=-1, learning_rate=0.04, min_child_samples=50,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1)))
    p["n_estimators"] = 6000
    p["verbose"] = -1
    if "subsample" in p:
        p.setdefault("subsample_freq", 1)
    Xl, Xl_test, cat_l = build_matrices(train_fe, test_fe, BASE_FEATURE_COLS, CAT_COLS, "native_cat")
    t0 = time.time()
    res = run_cv(lightgbm_factory(p, cat_l, seed=SEED, early_stopping_rounds=150), Xl, y, Xl_test, n_splits=N_SPLITS, seed=SEED)
    dt = time.time() - t0
    np.save(os.path.join(OOF_DIR, "final_lightgbm_oof.npy"), res["oof"])
    np.save(os.path.join(PRED_DIR, "final_lightgbm_test.npy"), res["test_pred"])
    log_experiment("final_lightgbm", "LightGBM", len(BASE_FEATURE_COLS), CAT_COLS, p, res["scores"], res["oof_auc"], res["fold_scores"], dt, f"STAGE B final, {N_SPLITS}-fold, tuned params, base features")
    summary.append(("lightgbm", res["oof_auc"], res["scores"]["mean"], res["scores"]["std"]))

# ---------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------
if "xgboost" in models_to_run:
    print(f"\n=== [FINAL] XGBoost  ({N_SPLITS}-fold) ===")
    p = dict(best_params.get("xgboost", {}).get("params", dict(
        max_depth=6, learning_rate=0.04, min_child_weight=5, subsample=0.8,
        colsample_bytree=0.8, gamma=0.0, reg_alpha=0.0, reg_lambda=1.0)))
    p["n_estimators"] = 6000
    Xx, Xx_test, cat_x = build_matrices(train_fe, test_fe, BASE_FEATURE_COLS, CAT_COLS, "native_cat")
    t0 = time.time()
    res = run_cv(xgboost_factory(p, seed=SEED, early_stopping_rounds=150), Xx, y, Xx_test, n_splits=N_SPLITS, seed=SEED)
    dt = time.time() - t0
    np.save(os.path.join(OOF_DIR, "final_xgboost_oof.npy"), res["oof"])
    np.save(os.path.join(PRED_DIR, "final_xgboost_test.npy"), res["test_pred"])
    log_experiment("final_xgboost", "XGBoost", len(BASE_FEATURE_COLS), CAT_COLS, p, res["scores"], res["oof_auc"], res["fold_scores"], dt, f"STAGE B final, {N_SPLITS}-fold, tuned params, base features")
    summary.append(("xgboost", res["oof_auc"], res["scores"]["mean"], res["scores"]["std"]))

# ---------------------------------------------------------------------------
# ExtraTrees (diversité) -- one-hot
# ---------------------------------------------------------------------------
if "extratrees" in models_to_run:
    print(f"\n=== [FINAL] ExtraTrees  ({N_SPLITS}-fold) ===")
    Xo, Xo_test, _ = build_matrices(train_fe, test_fe, BASE_FEATURE_COLS, CAT_COLS, "ohe")
    p = dict(n_estimators=800, max_depth=22, min_samples_leaf=3, max_features="sqrt")
    t0 = time.time()
    res = run_cv(extratrees_factory(p, seed=SEED), Xo, y, Xo_test, n_splits=N_SPLITS, seed=SEED)
    dt = time.time() - t0
    np.save(os.path.join(OOF_DIR, "final_extratrees_oof.npy"), res["oof"])
    np.save(os.path.join(PRED_DIR, "final_extratrees_test.npy"), res["test_pred"])
    log_experiment("final_extratrees", "ExtraTrees", Xo.shape[1], [], p, res["scores"], res["oof_auc"], res["fold_scores"], dt, f"STAGE B final, {N_SPLITS}-fold, OHE base features")
    summary.append(("extratrees", res["oof_auc"], res["scores"]["mean"], res["scores"]["std"]))

# ---------------------------------------------------------------------------
# HistGradientBoosting (diversité, natif catégoriel)
# ---------------------------------------------------------------------------
if "histgb" in models_to_run:
    print(f"\n=== [FINAL] HistGradientBoosting  ({N_SPLITS}-fold) ===")
    Xh, Xh_test, cat_h = build_matrices(train_fe, test_fe, BASE_FEATURE_COLS, CAT_COLS, "native_cat")
    p = dict(max_iter=3000, learning_rate=0.045, max_leaf_nodes=63, min_samples_leaf=40,
             l2_regularization=0.1, early_stopping=True, n_iter_no_change=60, validation_fraction=0.1)
    t0 = time.time()
    res = run_cv(histgb_factory(p, seed=SEED), Xh, y, Xh_test, n_splits=N_SPLITS, seed=SEED)
    dt = time.time() - t0
    np.save(os.path.join(OOF_DIR, "final_histgb_oof.npy"), res["oof"])
    np.save(os.path.join(PRED_DIR, "final_histgb_test.npy"), res["test_pred"])
    log_experiment("final_histgb", "HistGB", len(BASE_FEATURE_COLS), CAT_COLS, p, res["scores"], res["oof_auc"], res["fold_scores"], dt, f"STAGE B final, {N_SPLITS}-fold, base features")
    summary.append(("histgb", res["oof_auc"], res["scores"]["mean"], res["scores"]["std"]))

# ---------------------------------------------------------------------------
# RandomForest (diversité, sanity) -- one-hot
# ---------------------------------------------------------------------------
if "randomforest" in models_to_run:
    print(f"\n=== [FINAL] RandomForest  ({N_SPLITS}-fold) ===")
    Xo, Xo_test, _ = build_matrices(train_fe, test_fe, BASE_FEATURE_COLS, CAT_COLS, "ohe")
    p = dict(n_estimators=600, max_depth=20, min_samples_leaf=3, max_features="sqrt")
    t0 = time.time()
    res = run_cv(randomforest_factory(p, seed=SEED), Xo, y, Xo_test, n_splits=N_SPLITS, seed=SEED)
    dt = time.time() - t0
    np.save(os.path.join(OOF_DIR, "final_randomforest_oof.npy"), res["oof"])
    np.save(os.path.join(PRED_DIR, "final_randomforest_test.npy"), res["test_pred"])
    log_experiment("final_randomforest", "RandomForest", Xo.shape[1], [], p, res["scores"], res["oof_auc"], res["fold_scores"], dt, f"STAGE B final, {N_SPLITS}-fold, OHE base features")
    summary.append(("randomforest", res["oof_auc"], res["scores"]["mean"], res["scores"]["std"]))

# ---------------------------------------------------------------------------
# Logistic Regression (sanity baseline, modèle linéaire) -- one-hot + features enrichies
# ---------------------------------------------------------------------------
if "logreg" in models_to_run:
    print(f"\n=== [FINAL] LogisticRegression (sanity baseline)  ({N_SPLITS}-fold) ===")
    Xo, Xo_test, _ = build_matrices(train_fe, test_fe, ENGINEERED_FEATURE_COLS, CAT_COLS, "ohe")
    p = dict(max_iter=1000, C=1.0, solver="lbfgs")
    t0 = time.time()
    res = run_cv(logreg_factory(p, seed=SEED), Xo, y, Xo_test, n_splits=N_SPLITS, seed=SEED)
    dt = time.time() - t0
    np.save(os.path.join(OOF_DIR, "final_logreg_oof.npy"), res["oof"])
    np.save(os.path.join(PRED_DIR, "final_logreg_test.npy"), res["test_pred"])
    log_experiment("final_logreg", "LogReg", Xo.shape[1], [], p, res["scores"], res["oof_auc"], res["fold_scores"], dt, f"STAGE B final, {N_SPLITS}-fold, OHE+scaled enriched features (sanity baseline)")
    summary.append(("logreg", res["oof_auc"], res["scores"]["mean"], res["scores"]["std"]))

print("\n\n==================== STAGE B SUMMARY ====================")
for name, oof_auc, mean, std in summary:
    print(f"{name:15s} OOF AUC={oof_auc:.5f}  CV mean={mean:.5f}  std={std:.5f}")


