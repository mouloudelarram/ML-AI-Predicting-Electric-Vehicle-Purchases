"""
10_generate_experiment_log_md.py
Génère un EXPERIMENT_LOG.md lisible à la racine du projet à partir du CSV de log d'expériences.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from src.config import ROOT
from src.exp_log import LOG_CSV

out_path = os.path.join(ROOT, "EXPERIMENT_LOG.md")

if not os.path.exists(LOG_CSV):
    print("Aucun log d'expérience trouvé.")
    sys.exit(0)

df = pd.read_csv(LOG_CSV)
df = df.sort_values("cv_auc_mean", ascending=False).reset_index(drop=True)

lines = []
lines.append("# Journal d'expériences — Predicting Electric Vehicle Purchases\n")
lines.append(f"Total d'expériences enregistrées: **{len(df)}**\n")
lines.append("| Rang | Exp ID | Modèle | #Features | CV AUC (moyenne) | CV AUC (std) | OOF AUC | Notes |")
lines.append("|---|---|---|---|---|---|---|---|")
for i, row in df.iterrows():
    lines.append(
        f"| {i+1} | {row['exp_id']} | {row['model']} | {row['n_features']} | "
        f"{row['cv_auc_mean']:.5f} | {row['cv_auc_std']:.5f} | {row['oof_auc']:.5f} | {row['notes']} |"
    )

with open(out_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print(f"EXPERIMENT_LOG.md généré: {out_path} ({len(df)} expériences)")

