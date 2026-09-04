"""
08_make_submission.py
Génère le fichier de soumission final à partir de la meilleure config d'ensemble,
avec toutes les vérifications Kaggle-specific (shape, colonnes, unicité id, NaN, range).
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from src.config import ID_COL, TARGET_COL, SUB_DIR, PRED_DIR, LOG_DIR
from src.data import load_raw

_, test = load_raw()

test_pred_final = np.load(os.path.join(PRED_DIR, "ENSEMBLE_best_test.npy"))

with open(os.path.join(LOG_DIR, "ensemble_config.json")) as f:
    cfg = json.load(f)

print(f"Meilleure config utilisée: {cfg['best_key']}  (OOF AUC={cfg['best_oof_auc']:.5f})")

submission = pd.DataFrame({
    ID_COL: test[ID_COL].to_numpy(),
    TARGET_COL: test_pred_final,
})

# --- vérifications Kaggle ---
print("submission.shape       :", submission.shape)
print("submission.columns     :", list(submission.columns))
print("id unique              :", submission[ID_COL].is_unique)
print("row count == test rows :", len(submission) == len(test))
print("NaN values              :", submission.isnull().sum().sum())
print("prediction range        : [{:.6f}, {:.6f}]".format(submission[TARGET_COL].min(), submission[TARGET_COL].max()))
print("prediction mean          :", submission[TARGET_COL].mean())
print(submission[TARGET_COL].describe())

assert submission[ID_COL].is_unique, "IDs non uniques dans la soumission !"
assert len(submission) == len(test), "Nombre de lignes incorrect !"
assert submission.isnull().sum().sum() == 0, "Valeurs NaN détectées !"
assert submission[TARGET_COL].between(0, 1).all(), "Probabilités hors de [0,1] !"
assert (test[ID_COL].to_numpy() == submission[ID_COL].to_numpy()).all(), "Désalignement des IDs !"

out_path = os.path.join(SUB_DIR, "submission.csv")
submission.to_csv(out_path, index=False)
print(f"\nSoumission écrite: {out_path}")
print(submission.head())

