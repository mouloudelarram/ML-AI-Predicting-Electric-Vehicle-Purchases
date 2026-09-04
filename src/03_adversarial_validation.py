"""
03_adversarial_validation.py
Entraîne un classifieur train(0) vs test(1) pour détecter un éventuel drift de distribution.
Si l'AUC est proche de 0.5 -> train et test sont indistinguables (pas de drift).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

from src.config import NUM_COLS, CAT_COLS, ID_COL, TARGET_COL, SEED
from src.data import load_raw
from src.prep import align_categories

train, test = load_raw()
FEATURE_COLS = NUM_COLS + CAT_COLS + [ID_COL]

tr = train[FEATURE_COLS].copy()
te = test[FEATURE_COLS].copy()
tr["_is_test"] = 0
te["_is_test"] = 1
combined = pd.concat([tr, te], axis=0, ignore_index=True)
combined_no_id = combined.copy()

for c in CAT_COLS:
    combined_no_id[c] = combined_no_id[c].astype("category")

X = combined_no_id[NUM_COLS + CAT_COLS + [ID_COL]]
y = combined_no_id["_is_test"]

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
oof = np.zeros(len(X))
importances = np.zeros(X.shape[1])

for tr_idx, val_idx in skf.split(X, y):
    model = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, verbose=-1, random_state=SEED)
    model.fit(X.iloc[tr_idx], y.iloc[tr_idx], categorical_feature=CAT_COLS)
    oof[val_idx] = model.predict_proba(X.iloc[val_idx])[:, 1]
    importances += model.feature_importances_ / 5

auc_incl_id = roc_auc_score(y, oof)
print(f"Adversarial validation AUC (avec id)   : {auc_incl_id:.5f}  (0.5 = pas de drift)")

imp_series = pd.Series(importances, index=X.columns).sort_values(ascending=False)
print("\nImportance des features pour distinguer train/test:")
print(imp_series.to_string())

# refaire sans id pour isoler l'effet
X2 = combined_no_id[NUM_COLS + CAT_COLS]
oof2 = np.zeros(len(X2))
for tr_idx, val_idx in skf.split(X2, y):
    model = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, verbose=-1, random_state=SEED)
    model.fit(X2.iloc[tr_idx], y.iloc[tr_idx], categorical_feature=CAT_COLS)
    oof2[val_idx] = model.predict_proba(X2.iloc[val_idx])[:, 1]

auc_excl_id = roc_auc_score(y, oof2)
print(f"\nAdversarial validation AUC (sans id)  : {auc_excl_id:.5f}  (0.5 = pas de drift)")

with open("outputs/logs/03_adversarial_validation.txt", "w", encoding="utf-8") as f:
    f.write(f"AUC avec id: {auc_incl_id:.5f}\nAUC sans id: {auc_excl_id:.5f}\n\nImportances:\n{imp_series.to_string()}\n")
print("\nDone.")

