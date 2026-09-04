"""Préparation des matrices de features selon la famille de modèle."""
import pandas as pd


def to_category_dtype(df: pd.DataFrame, cat_cols) -> pd.DataFrame:
    df = df.copy()
    for c in cat_cols:
        df[c] = df[c].astype("category")
    return df


def align_categories(train_df: pd.DataFrame, test_df: pd.DataFrame, cat_cols):
    """S'assure que train/test partagent les mêmes catégories (utile pour lightgbm/xgboost/histgb)."""
    train_df = train_df.copy()
    test_df = test_df.copy()
    for c in cat_cols:
        cats = pd.api.types.union_categoricals(
            [train_df[c].astype("category"), test_df[c].astype("category")]
        ).categories
        train_df[c] = pd.Categorical(train_df[c], categories=cats)
        test_df[c] = pd.Categorical(test_df[c], categories=cats)
    return train_df, test_df


def one_hot_encode(train_df: pd.DataFrame, test_df: pd.DataFrame, cat_cols, other_cols):
    """One-hot sur cat_cols avec un fit combiné (train+test) pour garantir les mêmes colonnes."""
    n_train = len(train_df)
    combined = pd.concat([train_df[cat_cols + other_cols], test_df[cat_cols + other_cols]], axis=0, ignore_index=True)
    dummies = pd.get_dummies(combined, columns=cat_cols, drop_first=False)
    train_out = dummies.iloc[:n_train].reset_index(drop=True)
    test_out = dummies.iloc[n_train:].reset_index(drop=True)
    return train_out, test_out


def build_matrices(train_fe, test_fe, feature_cols, cat_cols, model_family):
    """
    model_family:
      - 'catboost'  -> DataFrame brut (str cols), cat_cols passé tel quel au modèle
      - 'native_cat'-> dtype 'category' pour cat_cols (lightgbm/xgboost/histgb)
      - 'ohe'       -> one-hot (extratrees/randomforest/logreg)
    Retourne (X_train, X_test, effective_cat_cols)
    """
    if model_family == "catboost":
        X_train = train_fe[feature_cols].copy()
        X_test = test_fe[feature_cols].copy()
        return X_train, X_test, cat_cols

    if model_family == "native_cat":
        num_cols = [c for c in feature_cols if c not in cat_cols]
        tr = train_fe[feature_cols].copy()
        te = test_fe[feature_cols].copy()
        tr, te = align_categories(tr, te, cat_cols)
        return tr, te, cat_cols

    if model_family == "ohe":
        num_cols = [c for c in feature_cols if c not in cat_cols]
        X_train, X_test = one_hot_encode(train_fe, test_fe, cat_cols, num_cols)
        return X_train, X_test, []

    raise ValueError(model_family)

