"""
Runner de validation croisée générique + fonctions d'entraînement par famille de modèle.

Chaque `*_fold_fn` a la signature:
    fn(X_tr, y_tr, X_val, y_val, X_test) -> (val_pred_proba, test_pred_proba)

`run_cv` orchestre le K-fold stratifié, agrège OOF + test (moyenne des folds), calcule les métriques.
"""
import time
import numpy as np
import pandas as pd
from src.cv_utils import get_folds, auc, summarize_scores


def run_cv(fold_fn_factory, X: pd.DataFrame, y: pd.Series, X_test: pd.DataFrame, n_splits=5, seed=42, verbose=True):
    folds = get_folds(y, n_splits=n_splits, seed=seed)
    oof = np.zeros(len(X))
    test_preds = np.zeros((n_splits, len(X_test)))
    fold_scores = []
    fold_times = []
    y_arr = y.to_numpy() if hasattr(y, "to_numpy") else np.asarray(y)

    for i, (tr_idx, val_idx) in enumerate(folds):
        t0 = time.time()
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
        fold_fn = fold_fn_factory(fold_i=i)
        val_pred, test_pred = fold_fn(X_tr, y_tr, X_val, y_val, X_test)
        oof[val_idx] = val_pred
        test_preds[i] = test_pred
        score = auc(y_val, val_pred)
        fold_scores.append(score)
        dt = time.time() - t0
        fold_times.append(dt)
        if verbose:
            print(f"    fold {i}: AUC={score:.5f}  time={dt:.1f}s")

    oof_auc = auc(y_arr, oof)
    scores = summarize_scores(fold_scores)
    return {
        "oof": oof,
        "test_pred": test_preds.mean(axis=0),
        "fold_scores": fold_scores,
        "oof_auc": oof_auc,
        "scores": scores,
        "total_time": sum(fold_times),
    }


# ----------------------------------------------------------------------------
# CatBoost
# ----------------------------------------------------------------------------
def catboost_factory(params, cat_cols, seed=42, early_stopping_rounds=100):
    from catboost import CatBoostClassifier

    def factory(fold_i):
        def _fn(X_tr, y_tr, X_val, y_val, X_test):
            p = dict(params)
            p.setdefault("random_seed", seed)
            model = CatBoostClassifier(
                **p,
                cat_features=cat_cols,
                eval_metric="AUC",
                loss_function="Logloss",
                early_stopping_rounds=early_stopping_rounds,
                verbose=False,
                allow_writing_files=False,
            )
            model.fit(X_tr, y_tr, eval_set=(X_val, y_val), use_best_model=True)
            return model.predict_proba(X_val)[:, 1], model.predict_proba(X_test)[:, 1]
        return _fn
    return factory


# ----------------------------------------------------------------------------
# LightGBM
# ----------------------------------------------------------------------------
def lightgbm_factory(params, cat_cols, seed=42, early_stopping_rounds=100):
    import lightgbm as lgb

    def factory(fold_i):
        def _fn(X_tr, y_tr, X_val, y_val, X_test):
            p = dict(params)
            p.setdefault("random_state", seed)
            model = lgb.LGBMClassifier(**p)
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                eval_metric="auc",
                categorical_feature=cat_cols,
                callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False), lgb.log_evaluation(0)],
            )
            return model.predict_proba(X_val)[:, 1], model.predict_proba(X_test)[:, 1]
        return _fn
    return factory


# ----------------------------------------------------------------------------
# XGBoost
# ----------------------------------------------------------------------------
def xgboost_factory(params, seed=42, early_stopping_rounds=100):
    import xgboost as xgb

    def factory(fold_i):
        def _fn(X_tr, y_tr, X_val, y_val, X_test):
            p = dict(params)
            p.setdefault("random_state", seed)
            model = xgb.XGBClassifier(
                **p,
                enable_categorical=True,
                tree_method="hist",
                eval_metric="auc",
                early_stopping_rounds=early_stopping_rounds,
            )
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            return model.predict_proba(X_val)[:, 1], model.predict_proba(X_test)[:, 1]
        return _fn
    return factory


# ----------------------------------------------------------------------------
# HistGradientBoosting (sklearn)
# ----------------------------------------------------------------------------
def histgb_factory(params, seed=42):
    from sklearn.ensemble import HistGradientBoostingClassifier

    def factory(fold_i):
        def _fn(X_tr, y_tr, X_val, y_val, X_test):
            p = dict(params)
            p.setdefault("random_state", seed)
            model = HistGradientBoostingClassifier(**p, categorical_features="from_dtype")
            model.fit(X_tr, y_tr)
            return model.predict_proba(X_val)[:, 1], model.predict_proba(X_test)[:, 1]
        return _fn
    return factory


# ----------------------------------------------------------------------------
# ExtraTrees / RandomForest (sklearn) -- attendent des features numériques (one-hot en amont)
# ----------------------------------------------------------------------------
def extratrees_factory(params, seed=42):
    from sklearn.ensemble import ExtraTreesClassifier

    def factory(fold_i):
        def _fn(X_tr, y_tr, X_val, y_val, X_test):
            p = dict(params)
            p.setdefault("random_state", seed)
            model = ExtraTreesClassifier(**p, n_jobs=-1)
            model.fit(X_tr, y_tr)
            return model.predict_proba(X_val)[:, 1], model.predict_proba(X_test)[:, 1]
        return _fn
    return factory


def randomforest_factory(params, seed=42):
    from sklearn.ensemble import RandomForestClassifier

    def factory(fold_i):
        def _fn(X_tr, y_tr, X_val, y_val, X_test):
            p = dict(params)
            p.setdefault("random_state", seed)
            model = RandomForestClassifier(**p, n_jobs=-1)
            model.fit(X_tr, y_tr)
            return model.predict_proba(X_val)[:, 1], model.predict_proba(X_test)[:, 1]
        return _fn
    return factory


# ----------------------------------------------------------------------------
# Logistic Regression (sanity baseline) -- attend des features standardisées + one-hot
# ----------------------------------------------------------------------------
def logreg_factory(params, seed=42):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    def factory(fold_i):
        def _fn(X_tr, y_tr, X_val, y_val, X_test):
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_val_s = scaler.transform(X_val)
            X_test_s = scaler.transform(X_test)
            p = dict(params)
            p.setdefault("random_state", seed)
            model = LogisticRegression(**p)
            model.fit(X_tr_s, y_tr)
            return model.predict_proba(X_val_s)[:, 1], model.predict_proba(X_test_s)[:, 1]
        return _fn
    return factory

