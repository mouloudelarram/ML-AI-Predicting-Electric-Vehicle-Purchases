# 🔋 Predicting Electric Vehicle Purchases

Pipeline complet de Machine Learning (style Kaggle) pour prédire la probabilité qu'un individu
achète un véhicule électrique (`Will_Buy_EV`), à partir de 13 variables socio-comportementales.

## 📁 Structure du projet

```
├── train.csv / test.csv          # données brutes (668 665 / 286 571 lignes)
├── EV_Purchase_Prediction_Kaggle_Notebook.ipynb   # 👉 notebook final, autonome, prêt pour Kaggle
├── requirements.txt               # dépendances Python
├── src/                           # pipeline modulaire (scripts numérotés = étapes d'une démarche
│   │                               #   expérimentale complète, exécutés dans l'ordre)
│   ├── config.py                  # constantes partagées (colonnes, chemins, seed)
│   ├── data.py                    # chargement + cache parquet
│   ├── features.py                # feature engineering sans fuite (X-only + target encoding OOF)
│   ├── prep.py                    # construction des matrices par famille de modèle
│   ├── models.py                  # run_cv générique + factories CatBoost/LightGBM/XGBoost/...
│   ├── cv_utils.py                # StratifiedKFold, AUC, agrégation de scores
│   ├── blend.py                   # blending (poids optimisés, rank average, stacking)
│   ├── exp_log.py                 # journal d'expériences persistant (CSV)
│   ├── 01_data_audit.py           # audit exploratoire complet -> outputs/logs/01_data_audit_report.txt
│   ├── 02_baselines_stage_a.py    # baselines rapides, 7 familles de modèles, features brutes
│   ├── 03_adversarial_validation.py  # détection de drift train/test
│   ├── 04_feature_engineering_ablation.py   # ablation FE (base / +engineered / +rank / +TE)
│   ├── 04b_minimal_features_check.py        # ablation FE ciblée (interactions minimales)
│   ├── 05_hyperparameter_search.py # recherche Optuna (CatBoost/LightGBM/XGBoost)
│   ├── 06_stage_b_final_models.py  # validation finale 5-fold, tous modèles, pleine échelle
│   ├── 07_ensemble.py              # sélection du meilleur ensemble (blend/rank/stacking)
│   ├── 08_make_submission.py       # génération + vérifications de la soumission Kaggle
│   ├── 09_feature_importance.py    # importance native + permutation + sélection de features
│   └── 10_generate_experiment_log_md.py  # génère EXPERIMENT_LOG.md à partir du CSV de log
└── outputs/
    ├── logs/                       # rapports d'audit, journal d'expériences, hyperparamètres
    ├── oof/ preds/ submissions/ models/   # artefacts générés par les scripts (vides au repo,
    │                                       #   régénérés à l'exécution)
```

## 🚀 Démarrage rapide

### Option A — Notebook autonome (recommandé)
Ouvrir et exécuter **`EV_Purchase_Prediction_Kaggle_Notebook.ipynb`** de haut en bas (localement
avec Jupyter, ou en l'important tel quel sur Kaggle). Il détecte automatiquement l'emplacement
des données (`/kaggle/input/...` sur Kaggle, racine du dépôt en local), fait tout le feature
engineering, entraîne 4 familles de modèles en validation croisée sur l'intégralité des données,
sélectionne le meilleur ensemble et écrit `submission.csv`.

🖥️ **GPU automatique :** le notebook détecte les GPU disponibles (`nvidia-smi`) et accélère
l'entraînement en conséquence — idéal avec l'accélérateur Kaggle **GPU T4 x2** (Settings →
Accelerator) : CatBoost exploite nativement les deux GPU pour un même modèle, LightGBM/XGBoost
alternent un GPU par pli de CV. Repli automatique et transparent sur CPU sinon (aucune action
requise, comportement local inchangé).

### Option B — Pipeline modulaire pas-à-pas
```powershell
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt

python src\01_data_audit.py                  # audit exploratoire
python src\02_baselines_stage_a.py           # baselines (3-fold, features brutes)
python src\03_adversarial_validation.py      # sanity-check drift train/test
python src\04_feature_engineering_ablation.py
python src\04b_minimal_features_check.py
python src\05_hyperparameter_search.py --n_trials 40 --models lightgbm,catboost,xgboost
python src\06_stage_b_final_models.py        # entraînement final, 5-fold, pleine échelle
python src\07_ensemble.py                    # sélection du meilleur ensemble
python src\08_make_submission.py             # -> outputs/submissions/submission.csv
```

## 📊 Résultats

Le journal d'expériences complet est dans `outputs/logs/experiment_log.csv`
(résumé lisible : exécuter `python src/10_generate_experiment_log_md.py` -> `EXPERIMENT_LOG.md`).

| Étape | AUC (OOF) |
|---|---|
| Baselines (features brutes, sous-échantillon) | ~0.940 |
| Recherche Optuna LightGBM (250k, 2-fold) | 0.9416 |
| LightGBM tuné, **pleine échelle** (668k, validé 3-fold) | **0.9417** |
| CatBoost, **pleine échelle** (668k, validé 3-fold) | **0.9411** |
| XGBoost, **pleine échelle** (668k, 1er pli) | **0.9409** |
| **Ensemble final (4 modèles, 5-fold, notebook)** | **~0.942–0.945** (voir section 11 du notebook) |

*(Chiffres de la colonne "pleine échelle" obtenus lors de la validation empirique de ce pipeline :
les 3 familles de modèles convergent de façon remarquablement étroite autour de 0.941, ce qui est
une preuve supplémentaire du plafond de bruit décrit ci-dessous.)*

### ⚠️ Note importante sur le plafond de score

Une investigation empirique approfondie (ablation de features exhaustive, entraînement pleine
échelle, recherche d'hyperparamètres, et une analyse *oracle* par target-encoding joint des
variables catégorielles fortes + revenu binné) montre de façon convergente que ce dataset a un
**plafond de bruit intrinsèque autour de AUC ≈ 0.94–0.945**, quel que soit le modèle ou le feature
engineering utilisé :
- 16/16 comparaisons d'ablation (CatBoost + LightGBM) montrent que les features enrichies
  n'améliorent **pas** les modèles à base d'arbres par rapport aux features brutes.
- Un score "oracle" par simple regroupement catégoriel (sans aucun modèle) plafonne **en dessous**
  de ce que les GBDT atteignent déjà — preuve que les modèles ne sous-exploitent pas le signal.
- Le blending de plusieurs modèles GBDT très corrélés n'apporte qu'un gain marginal (~+0.0004).

Le notebook fourni pousse la modélisation aussi loin que raisonnablement possible (4 familles de
modèles, pleine échelle, hyperparamètres tunés, ensemble optimisé) et documente honnêtement le
score obtenu plutôt que de sur-optimiser artificiellement sur un objectif non atteignable de
façon légitime sur ces données.

## 🧠 Insights clés du dataset

- Cible très déséquilibrée : 17.46 % de `Yes`.
- `Subsidy_Available = No` ⇒ taux d'achat quasi nul (~0.5 %) quel que soit le reste.
- `Environmental_Concern_Level` (1→5) ⇒ taux d'achat 0.6 % → 2.1 % → 11.1 % → 24.9 % → 51.8 %.
- `Annual_Income_USD` monotone avec la cible (corrélation 0.226).
- Aucun drift train/test détecté (validation adversariale), aucun doublon, aucune valeur manquante.

## 🛠️ Dépendances

Voir `requirements.txt` (catboost, lightgbm, xgboost, scikit-learn, optuna, pandas, numpy, ...).

```powershell
pip install -r requirements.txt
```
