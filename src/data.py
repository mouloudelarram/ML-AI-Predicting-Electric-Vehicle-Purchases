"""Chargement + cache des données (parquet pour accélérer les itérations)."""
import os
import pandas as pd
from src.config import TRAIN_PATH, TEST_PATH, ROOT

TRAIN_PARQUET = os.path.join(ROOT, "outputs", "train_cache.parquet")
TEST_PARQUET = os.path.join(ROOT, "outputs", "test_cache.parquet")


def load_raw(use_cache=True):
    if use_cache and os.path.exists(TRAIN_PARQUET) and os.path.exists(TEST_PARQUET):
        train = pd.read_parquet(TRAIN_PARQUET)
        test = pd.read_parquet(TEST_PARQUET)
    else:
        train = pd.read_csv(TRAIN_PATH)
        test = pd.read_csv(TEST_PATH)
        # normaliser dtype str -> object pour compat maximale avec toutes les libs
        for c in train.select_dtypes(include="object").columns:
            train[c] = train[c].astype(str)
        for c in test.select_dtypes(include="object").columns:
            test[c] = test[c].astype(str)
        for c in train.columns:
            if train[c].dtype == "string" or str(train[c].dtype) == "str":
                train[c] = train[c].astype(str)
        for c in test.columns:
            if test[c].dtype == "string" or str(test[c].dtype) == "str":
                test[c] = test[c].astype(str)
        train.to_parquet(TRAIN_PARQUET, index=False)
        test.to_parquet(TEST_PARQUET, index=False)
    return train, test

