"""
07_ensemble.py
Charge les OOF/test predictions "final_<model>", calcule:
 - AUC individuel de chaque modèle
 - ensemble par moyenne pondérée de probabilités (poids optimisés sur OOF)
 - ensemble par moyenne de rangs (poids optimisés sur OOF)
 - stacking (Logistic Regression sur OOF, validé en CV)
Sélectionne le meilleur set-up et sauvegarde sa config pour la génération de la soumission finale.
"""
import sys, os, json, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.config import TARGET_COL, SEED, OOF_DIR, PRED_DIR, LOG_DIR
from src.data import load_raw
from src.blend import optimize_weights_random_search, rank_transform, blend_predictions, stacking_cv, apply_stacking_to_test

train, test = load_raw()
y = train[TARGET_COL].map({"Yes": 1, "No": 0}).astype(int)

oof_files = sorted(glob.glob(os.path.join(OOF_DIR, "final_*_oof.npy")))
model_names = [os.path.basename(f).replace("final_", "").replace("_oof.npy", "") for f in oof_files]
print(f"Modèles finaux détectés: {model_names}")

oof_dict, test_dict = {}, {}
for name in model_names:
    oof_dict[name] = np.load(os.path.join(OOF_DIR, f"final_{name}_oof.npy"))
    test_dict[name] = np.load(os.path.join(PRED_DIR, f"final_{name}_test.npy"))

print("\n--- AUC individuel (OOF) ---")
individual_aucs = {}
for name in model_names:
    a = roc_auc_score(y, oof_dict[name])
    individual_aucs[name] = a
    print(f"  {name:15s}: {a:.5f}")

results = {}

# 1) simple average (equal weight) baseline
equal_w = {n: 1.0 / len(model_names) for n in model_names}
equal_pred = blend_predictions(oof_dict, equal_w)
equal_auc = roc_auc_score(y, equal_pred)
print(f"\n[Equal-weight probability average] AUC = {equal_auc:.5f}")
results["equal_avg"] = {"auc": equal_auc, "weights": equal_w, "type": "prob"}

# 2) pairwise classic grid (0.5/0.5, 0.7/0.3, ...) if exactly matches common pairs -- informational
if len(model_names) >= 2:
    print("\n--- Grille classique pour les 2 meilleurs modèles individuels ---")
    top2 = sorted(individual_aucs, key=individual_aucs.get, reverse=True)[:2]
    for w1 in [0.5, 0.6, 0.7, 0.8, 0.3, 0.4, 0.2]:
        w = {top2[0]: w1, top2[1]: 1 - w1}
        pred = blend_predictions({k: oof_dict[k] for k in top2}, w)
        a = roc_auc_score(y, pred)
        print(f"   {top2[0]}={w1:.1f} / {top2[1]}={1-w1:.1f}  -> AUC={a:.5f}")

# 3) optimized probability weights (full model set)
print("\n--- Optimisation des poids (probabilités, recherche Dirichlet + raffinement) ---")
opt_w, opt_auc = optimize_weights_random_search(oof_dict, y, n_iter=4000, seed=SEED)
print(f"Poids optimaux: { {k: round(v,4) for k,v in opt_w.items()} }")
print(f"AUC (probability blend optimisé) = {opt_auc:.5f}")
results["prob_blend_optimized"] = {"auc": opt_auc, "weights": opt_w, "type": "prob"}

# 4) rank averaging
print("\n--- Rank averaging ---")
oof_ranks = rank_transform(oof_dict)
rank_equal_pred = blend_predictions(oof_ranks, equal_w)
rank_equal_auc = roc_auc_score(y, rank_equal_pred)
print(f"[Equal-weight rank average] AUC = {rank_equal_auc:.5f}")

rank_opt_w, rank_opt_auc = optimize_weights_random_search(oof_ranks, y, n_iter=4000, seed=SEED)
print(f"Poids optimaux (rank): { {k: round(v,4) for k,v in rank_opt_w.items()} }")
print(f"AUC (rank blend optimisé) = {rank_opt_auc:.5f}")
results["rank_blend_optimized"] = {"auc": rank_opt_auc, "weights": rank_opt_w, "type": "rank"}
results["rank_blend_equal"] = {"auc": rank_equal_auc, "weights": equal_w, "type": "rank"}

# 5) stacking
print("\n--- Stacking (Logistic Regression meta-model, CV) ---")
for C in [0.1, 1.0, 10.0]:
    meta_oof, meta_auc, final_meta, names = stacking_cv(oof_dict, y, n_splits=5, seed=SEED, C=C)
    print(f"  C={C:<5} stacking OOF AUC = {meta_auc:.5f}")
    results[f"stacking_C{C}"] = {"auc": meta_auc, "type": "stacking", "C": C}

# pick best C properly and keep model
best_stack_C = max([0.1, 1.0, 10.0], key=lambda c: results[f"stacking_C{c}"]["auc"])
meta_oof, meta_auc, final_meta, names = stacking_cv(oof_dict, y, n_splits=5, seed=SEED, C=best_stack_C)

print("\n==================== RÉSUMÉ ENSEMBLE ====================")
for k, v in sorted(results.items(), key=lambda kv: kv[1]["auc"], reverse=True):
    print(f"  {k:28s} AUC={v['auc']:.5f}")

best_key = max(results, key=lambda k: results[k]["auc"])
print(f"\n>>> Meilleure config: {best_key}  AUC OOF = {results[best_key]['auc']:.5f}")

# --- construire la prédiction test correspondant à la meilleure config ---
if results[best_key]["type"] == "prob":
    test_pred_final = blend_predictions(test_dict, results[best_key]["weights"])
elif results[best_key]["type"] == "rank":
    test_ranks = rank_transform(test_dict)
    test_pred_final = blend_predictions(test_ranks, results[best_key]["weights"])
elif results[best_key]["type"] == "stacking":
    test_pred_final = apply_stacking_to_test(final_meta, test_dict, names)
else:
    raise ValueError(results[best_key]["type"])

np.save(os.path.join(PRED_DIR, "ENSEMBLE_best_test.npy"), test_pred_final)
np.save(os.path.join(OOF_DIR, "ENSEMBLE_best_oof.npy"),
        blend_predictions(oof_dict, results[best_key]["weights"]) if results[best_key]["type"] == "prob" else
        (blend_predictions(rank_transform(oof_dict), results[best_key]["weights"]) if results[best_key]["type"] == "rank" else meta_oof))

config_to_save = {
    "best_key": best_key,
    "best_oof_auc": results[best_key]["auc"],
    "individual_aucs": individual_aucs,
    "all_results": {k: {kk: vv for kk, vv in v.items() if kk != "weights"} | ({"weights": v["weights"]} if "weights" in v else {}) for k, v in results.items()},
    "model_names": model_names,
}
with open(os.path.join(LOG_DIR, "ensemble_config.json"), "w") as f:
    json.dump(config_to_save, f, indent=2, default=str)
print(f"\nConfig ensemble sauvegardée dans {os.path.join(LOG_DIR, 'ensemble_config.json')}")

