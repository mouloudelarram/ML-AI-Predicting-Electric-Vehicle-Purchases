"""Gestion du log d'expériences (CSV persistant + impression console)."""
import os
import json
import time
import pandas as pd
from src.config import LOG_DIR

LOG_CSV = os.path.join(LOG_DIR, "experiment_log.csv")

COLUMNS = [
    "timestamp", "exp_id", "model", "n_features", "cat_cols", "params",
    "cv_auc_mean", "cv_auc_std", "cv_auc_min", "cv_auc_max", "oof_auc",
    "fold_scores", "train_time_sec", "notes",
]


def log_experiment(exp_id, model, n_features, cat_cols, params, scores, oof_auc, fold_scores, train_time_sec, notes=""):
    row = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "exp_id": exp_id,
        "model": model,
        "n_features": n_features,
        "cat_cols": json.dumps(cat_cols),
        "params": json.dumps(params, default=str),
        "cv_auc_mean": scores["mean"],
        "cv_auc_std": scores["std"],
        "cv_auc_min": scores["min"],
        "cv_auc_max": scores["max"],
        "oof_auc": oof_auc,
        "fold_scores": json.dumps([round(float(s), 6) for s in fold_scores]),
        "train_time_sec": train_time_sec,
        "notes": notes,
    }
    if os.path.exists(LOG_CSV):
        df = pd.read_csv(LOG_CSV)
        df = pd.concat([df, pd.DataFrame([row])], axis=0, ignore_index=True)
    else:
        df = pd.DataFrame([row], columns=COLUMNS)
    df.to_csv(LOG_CSV, index=False)
    print(f"[LOG] {exp_id} | {model} | CV AUC mean={scores['mean']:.5f} std={scores['std']:.5f} "
          f"min={scores['min']:.5f} max={scores['max']:.5f} | OOF AUC={oof_auc:.5f} | notes={notes}")
    return row


def show_log(top=30):
    if not os.path.exists(LOG_CSV):
        print("No experiment log yet.")
        return
    df = pd.read_csv(LOG_CSV)
    df_sorted = df.sort_values("cv_auc_mean", ascending=False)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print(df_sorted[["exp_id", "model", "n_features", "cv_auc_mean", "cv_auc_std", "oof_auc", "notes"]].head(top).to_string(index=False))

