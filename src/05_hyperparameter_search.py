"""
05_hyperparameter_search.py
Recherche d'hyperparamètres avec Optuna pour CatBoost, LightGBM, XGBoost.
Stratégie "Stage A rapide": sous-échantillon stratifié + CV à 2 folds + itérations réduites.
Les meilleurs paramètres sont sauvegardés en JSON pour la validation finale 5-fold (Stage B).
"""
import sys, os, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import train_test_split

from src.config import NUM_COLS, CAT_COLS, TARGET_COL, SEED, LOG_DIR
from src.data import load_raw
from src.features import add_engineered_features, add_rank_features, ENGINEERED_COLS
from src.prep import build_matrices
from src.cv_utils import get_folds, auc, summarize_scores
from src.models import catboost_factory, lightgbm_factory, xgboost_factory, run_cv

optuna.logging.set_verbosity(optuna.logging.WARNING)

parser = argparse.ArgumentParser()
parser.add_argument("--n_trials", type=int, default=25)
parser.add_argument("--subsample_n", type=int, default=200000)
parser.add_argument("--n_splits", type=int, default=2)
parser.add_argument("--models", type=str, default="catboost,lightgbm,xgboost")
args = parser.parse_args()

print("Loading + engineering features...")
train, test = load_raw()
y_full = train[TARGET_COL].map({"Yes": 1, "No": 0}).astype(int)
train_fe = add_engineered_features(train)
test_fe = add_engineered_features(test)
rank_cols = ["Age", "Annual_Income_USD", "Daily_Commute_km"]
train_fe, test_fe = add_rank_features(train_fe, test_fe, rank_cols)
# NOTE: l'ablation (04_feature_engineering_ablation.py + 04b_minimal_features_check.py) a montré
# de façon cohérente (16/16 comparaisons, CatBoost+LightGBM) que TOUTES les variantes de feature
# engineering testées sont légèrement INFÉRIEURES aux features brutes pour les modèles arbres
# (CatBoost/LightGBM capturent déjà nativement les interactions catégorielles à faible cardinalité
# via des splits séquentiels). On utilise donc le set de base pour la recherche d'hyperparamètres.
FEATURE_COLS = NUM_COLS + CAT_COLS

# sous-échantillon stratifié pour accélérer la recherche
if args.subsample_n < len(train_fe):
    idx_sub, _ = train_test_split(
        np.arange(len(train_fe)), train_size=args.subsample_n, stratify=y_full, random_state=SEED
    )
    train_sub = train_fe.iloc[idx_sub].reset_index(drop=True)
    y_sub = y_full.iloc[idx_sub].reset_index(drop=True)
else:
    train_sub = train_fe
    y_sub = y_full
print(f"Search subsample size: {len(train_sub)}  (full train={len(train_fe)})")

best_params_path = os.path.join(LOG_DIR, "best_params.json")
best_params_all = {}
if os.path.exists(best_params_path):
    with open(best_params_path) as f:
        best_params_all = json.load(f)


def objective_catboost(trial):
    params = dict(
        iterations=1200,
        depth=trial.suggest_int("depth", 4, 9),
        learning_rate=trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
        l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 12.0),
        random_strength=trial.suggest_float("random_strength", 0.0, 4.0),
        bagging_temperature=trial.suggest_float("bagging_temperature", 0.0, 3.0),
        border_count=trial.suggest_categorical("border_count", [32, 64, 128, 254]),
        max_ctr_complexity=2,
    )
    Xc, _, cat_c = build_matrices(train_sub, test_fe, FEATURE_COLS, CAT_COLS, "catboost")
    res = run_cv(catboost_factory(params, cat_c, seed=SEED, early_stopping_rounds=50), Xc, y_sub, Xc.iloc[:100],
                 n_splits=args.n_splits, seed=SEED, verbose=False)
    trial.set_user_attr("fold_scores", res["fold_scores"])
    return res["scores"]["mean"]


def objective_lightgbm(trial):
    params = dict(
        n_estimators=1500,
        num_leaves=trial.suggest_int("num_leaves", 16, 255, log=True),
        max_depth=trial.suggest_categorical("max_depth", [-1, 4, 6, 8, 10]),
        learning_rate=trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
        min_child_samples=trial.suggest_int("min_child_samples", 5, 200, log=True),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        subsample_freq=1,
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 5.0, log=True),
        verbose=-1,
    )
    Xl, _, cat_l = build_matrices(train_sub, test_fe, FEATURE_COLS, CAT_COLS, "native_cat")
    res = run_cv(lightgbm_factory(params, cat_l, seed=SEED, early_stopping_rounds=60), Xl, y_sub, Xl.iloc[:100],
                 n_splits=args.n_splits, seed=SEED, verbose=False)
    trial.set_user_attr("fold_scores", res["fold_scores"])
    return res["scores"]["mean"]


def objective_xgboost(trial):
    params = dict(
        n_estimators=1500,
        max_depth=trial.suggest_int("max_depth", 3, 10),
        learning_rate=trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
        min_child_weight=trial.suggest_float("min_child_weight", 1.0, 20.0, log=True),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        gamma=trial.suggest_float("gamma", 1e-3, 5.0, log=True),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 0.5, 5.0, log=True),
    )
    Xx, _, cat_x = build_matrices(train_sub, test_fe, FEATURE_COLS, CAT_COLS, "native_cat")
    res = run_cv(xgboost_factory(params, seed=SEED, early_stopping_rounds=60), Xx, y_sub, Xx.iloc[:100],
                 n_splits=args.n_splits, seed=SEED, verbose=False)
    trial.set_user_attr("fold_scores", res["fold_scores"])
    return res["scores"]["mean"]


OBJECTIVES = {
    "catboost": objective_catboost,
    "lightgbm": objective_lightgbm,
    "xgboost": objective_xgboost,
}

models_to_run = args.models.split(",")
for model_name in models_to_run:
    print(f"\n{'='*70}\nOptuna search: {model_name}  (n_trials={args.n_trials}, n_splits={args.n_splits})\n{'='*70}")
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    t0 = time.time()
    study.optimize(OBJECTIVES[model_name], n_trials=args.n_trials, show_progress_bar=False)
    dt = time.time() - t0
    print(f"Best {model_name} CV AUC (subsample, {args.n_splits}-fold): {study.best_value:.5f}  time={dt:.1f}s")
    print(f"Best params: {study.best_params}")
    best_params_all[model_name] = {"params": study.best_params, "cv_auc_subsample": study.best_value}

    trials_df = study.trials_dataframe()
    trials_df.to_csv(os.path.join(LOG_DIR, f"optuna_trials_{model_name}.csv"), index=False)

with open(best_params_path, "w") as f:
    json.dump(best_params_all, f, indent=2)
print(f"\nSaved best params to {best_params_path}")


