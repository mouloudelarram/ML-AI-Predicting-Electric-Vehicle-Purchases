"""
01_data_audit.py
Audit complet du dataset train/test :
- shape, dtypes, valeurs manquantes
- doublons (lignes, ids)
- distribution de la cible
- cardinalité catégorielle / valeurs uniques
- distributions numériques, outliers
- comparaison train/test (moyenne, std, quantiles)
- taux de cible par catégorie
- taux de cible par bin numérique
- corrélations
- analyse de la colonne id (ordre, fuite potentielle)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from src.config import TRAIN_PATH, TEST_PATH, ID_COL, TARGET_COL, CAT_COLS, NUM_COLS, LOG_DIR

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 50)

report_lines = []

def log(msg=""):
    print(msg)
    report_lines.append(str(msg))

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

log("=" * 80)
log("1. SHAPE")
log("=" * 80)
log(f"train shape: {train.shape}")
log(f"test shape : {test.shape}")
log(f"train columns: {list(train.columns)}")
log(f"test columns : {list(test.columns)}")

log("\n" + "=" * 80)
log("2. DTYPES")
log("=" * 80)
log(train.dtypes.to_string())

log("\n" + "=" * 80)
log("3. MISSING VALUES")
log("=" * 80)
log("train missing:\n" + train.isnull().sum().to_string())
log("\ntest missing:\n" + test.isnull().sum().to_string())

log("\n" + "=" * 80)
log("4. DUPLICATES")
log("=" * 80)
log(f"train duplicated full rows (incl id): {train.duplicated().sum()}")
log(f"train duplicated rows (excl id): {train.drop(columns=[ID_COL]).duplicated().sum()}")
log(f"train duplicated ids: {train[ID_COL].duplicated().sum()}")
log(f"test duplicated rows (excl id): {test.drop(columns=[ID_COL]).duplicated().sum()}")
log(f"test duplicated ids: {test[ID_COL].duplicated().sum()}")
overlap_ids = set(train[ID_COL]).intersection(set(test[ID_COL]))
log(f"train/test id overlap count: {len(overlap_ids)}")
# feature-only duplicate overlap between train and test (excluding id and target)
feat_cols = [c for c in train.columns if c not in [ID_COL, TARGET_COL]]
train_feat = train[feat_cols].copy()
test_feat = test[feat_cols].copy()
train_feat["_src"] = "train"
test_feat["_src"] = "test"
combo = pd.concat([train_feat, test_feat], axis=0)
dup_mask = combo.drop(columns=["_src"]).duplicated(keep=False)
log(f"rows in train+test sharing an identical feature-combination with another row: {dup_mask.sum()} "
    f"({dup_mask.sum() / len(combo):.4%})")

log("\n" + "=" * 80)
log("5. TARGET DISTRIBUTION")
log("=" * 80)
log(train[TARGET_COL].value_counts().to_string())
log(train[TARGET_COL].value_counts(normalize=True).to_string())
target_map = {"Yes": 1, "No": 0}
train["_target_num"] = train[TARGET_COL].map(target_map)
log(f"positive rate: {train['_target_num'].mean():.4f}")

log("\n" + "=" * 80)
log("6. NUMERICAL DESCRIBE (train vs test)")
log("=" * 80)
for c in NUM_COLS:
    log(f"\n--- {c} ---")
    tr_desc = train[c].describe(percentiles=[.01,.05,.25,.5,.75,.95,.99])
    te_desc = test[c].describe(percentiles=[.01,.05,.25,.5,.75,.95,.99])
    cmp_df = pd.DataFrame({"train": tr_desc, "test": te_desc})
    log(cmp_df.to_string())
    log(f"n_unique train: {train[c].nunique()}  | n_unique test: {test[c].nunique()}")

log("\n" + "=" * 80)
log("7. CATEGORICAL CARDINALITY & DISTRIBUTION (train vs test)")
log("=" * 80)
for c in CAT_COLS:
    log(f"\n--- {c} ---")
    tr_vc = train[c].value_counts(normalize=True).rename("train_pct")
    te_vc = test[c].value_counts(normalize=True).rename("test_pct")
    cmp_df = pd.concat([tr_vc, te_vc], axis=1)
    log(cmp_df.to_string())

log("\n" + "=" * 80)
log("8. TARGET RATE BY CATEGORY")
log("=" * 80)
for c in CAT_COLS:
    log(f"\n--- {c} ---")
    grp = train.groupby(c)["_target_num"].agg(["mean", "count"])
    log(grp.to_string())

log("\n" + "=" * 80)
log("9. TARGET RATE BY NUMERICAL BIN (quantile bins, q=10)")
log("=" * 80)
for c in NUM_COLS:
    log(f"\n--- {c} ---")
    try:
        bins = pd.qcut(train[c], q=10, duplicates="drop")
        grp = train.groupby(bins, observed=True)["_target_num"].agg(["mean", "count"])
        log(grp.to_string())
    except Exception as e:
        log(f"qcut failed: {e}")

log("\n" + "=" * 80)
log("10. TARGET RATE BY DISCRETE INTEGER VALUE (low cardinality numeric)")
log("=" * 80)
for c in ["Number_of_Cars_Owned", "Charging_Stations_Near_Home", "Charging_Stations_Near_Work", "Environmental_Concern_Level"]:
    log(f"\n--- {c} ---")
    grp = train.groupby(c)["_target_num"].agg(["mean", "count"])
    log(grp.to_string())

log("\n" + "=" * 80)
log("11. TWO-WAY INTERACTIONS: TARGET RATE (selected combos)")
log("=" * 80)
combo_pairs = [
    ("Home_Charging_Possible", "Subsidy_Available"),
    ("Home_Charging_Possible", "Range_Anxiety_Level"),
    ("Subsidy_Available", "Range_Anxiety_Level"),
    ("City_Type", "Home_Charging_Possible"),
    ("Current_Car_Type", "Range_Anxiety_Level"),
]
for a, b in combo_pairs:
    log(f"\n--- {a} x {b} ---")
    grp = train.groupby([a, b])["_target_num"].agg(["mean", "count"])
    log(grp.to_string())

log("\n" + "=" * 80)
log("12. CORRELATIONS (numeric + target)")
log("=" * 80)
num_plus_target = NUM_COLS + ["_target_num"]
corr = train[num_plus_target].corr()
log(corr.to_string())

log("\n" + "=" * 80)
log("13. ID ANALYSIS")
log("=" * 80)
log(f"train id min/max: {train[ID_COL].min()} / {train[ID_COL].max()}")
log(f"test id min/max : {test[ID_COL].min()} / {test[ID_COL].max()}")
log(f"train id is contiguous range: {(train[ID_COL].sort_values().reset_index(drop=True) == pd.Series(range(train[ID_COL].min(), train[ID_COL].min()+len(train)))).all()}")
# correlation between id and target
log(f"corr(id, target): {train[[ID_COL,'_target_num']].corr().iloc[0,1]:.5f}")
# target rate by id decile
id_bins = pd.qcut(train[ID_COL], q=20, duplicates="drop")
grp = train.groupby(id_bins, observed=True)["_target_num"].agg(["mean", "count"])
log("target rate by id-vigntile:\n" + grp.to_string())

log("\n" + "=" * 80)
log("14. SUSPICIOUS VALUES / OUTLIERS")
log("=" * 80)
for c in NUM_COLS:
    q1, q99 = train[c].quantile([0.01, 0.99])
    n_low = (train[c] < q1).sum()
    n_high = (train[c] > q99).sum()
    log(f"{c}: min={train[c].min()}, max={train[c].max()}, <1%={n_low}, >99%={n_high}")

log("\n" + "=" * 80)
log("15. EXACT DUPLICATE FEATURE-ROWS WITH CONFLICTING TARGETS (train only)")
log("=" * 80)
feat_no_id = [c for c in train.columns if c not in [ID_COL, TARGET_COL, "_target_num"]]
dup_groups = train.groupby(feat_no_id)["_target_num"].agg(["mean", "count", "nunique"])
conflict = dup_groups[(dup_groups["count"] > 1) & (dup_groups["nunique"] > 1)]
log(f"number of feature-combinations appearing >1 time: {(dup_groups['count']>1).sum()}")
log(f"number of feature-combinations with conflicting targets: {len(conflict)}")
if len(conflict) > 0:
    log(conflict.head(20).to_string())

out_path = os.path.join(LOG_DIR, "01_data_audit_report.txt")
with open(out_path, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))
print(f"\n\nReport written to {out_path}")

