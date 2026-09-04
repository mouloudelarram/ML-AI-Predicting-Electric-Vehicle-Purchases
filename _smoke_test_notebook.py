"""Smoke test rapide: extrait le code de chaque cellule du notebook et l'execute
sur un PETIT sous-echantillon (5000 lignes) avec des hyperparametres reduits, pour
detecter d'eventuelles erreurs runtime (API, noms de variables, etc.) sans attendre
l'entrainement complet. Remplace train.csv/test.csv par des versions reduites en memoire
via monkeypatching de pd.read_csv le temps du test.
"""
import json, sys, os, types
import pandas as pd
import numpy as np

nb = json.load(open("EV_Purchase_Prediction_Kaggle_Notebook.ipynb", encoding="utf-8"))
code_cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
full_src = "\n\n".join(code_cells)

# --- patchs pour un smoke test rapide ---
full_src = full_src.replace("N_FOLDS = 5", "N_FOLDS = 2")
full_src = full_src.replace("n_estimators=6000", "n_estimators=80")
full_src = full_src.replace("iterations=6000", "iterations=80")
full_src = full_src.replace("max_iter=2500", "max_iter=80")
full_src = full_src.replace("early_stopping_rounds=150", "early_stopping_rounds=20")
full_src = full_src.replace("n_iter_no_change=80", "n_iter_no_change=20")
full_src = full_src.replace("n_iter=4000", "n_iter=50")
full_src = full_src.replace("refine_rounds=3", "refine_rounds=1")
full_src = full_src.replace('submission.to_csv("submission.csv", index=False)',
                             'submission.to_csv("outputs/submissions/_smoketest_submission.csv", index=False)')

# limiter la taille des donnees chargees (sans toucher aux fichiers reels)
_orig_read_csv = pd.read_csv
def _patched_read_csv(path, *a, **kw):
    df = _orig_read_csv(path, *a, **kw)
    if "train.csv" in str(path) or "test.csv" in str(path):
        return df.sample(n=min(5000, len(df)), random_state=42).reset_index(drop=True)
    return df
pd.read_csv = _patched_read_csv

g = {"__name__": "__main__"}
try:
    exec(compile(full_src, "<notebook>", "exec"), g)
    print("\n\n>>> SMOKE TEST OK: le notebook s'execute sans erreur de bout en bout.")
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"\n\n>>> SMOKE TEST FAILED: {e}")
    sys.exit(1)

