"""
Feature engineering réutilisable et sans fuite (leakage-free) pour le pipeline EV Purchase.

Toutes les fonctions ici sont "X-only" (n'utilisent jamais la cible), donc safe à appliquer
indépendamment sur train et test, ou sur train+test concaténés (pour les rangs percentiles).

L'encodage cible (target encoding) est traité à part car il DOIT être out-of-fold (voir target_encode_oof).
"""
import numpy as np
import pandas as pd

RANGE_ANXIETY_MAP = {"Low": 0, "Medium": 1, "High": 2}
CITY_DENSITY_MAP = {"Urban": 2, "Suburban": 1, "Rural": 0}


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute des features dérivées non-linéaires, ratios, interactions. Ne modifie pas les colonnes brutes."""
    df = df.copy()

    # --- ordinal / binary encodings (features numériques additionnelles, on garde les colonnes brutes) ---
    df["Range_Anxiety_Ordinal"] = df["Range_Anxiety_Level"].map(RANGE_ANXIETY_MAP).astype(float)
    df["Home_Charging_Binary"] = (df["Home_Charging_Possible"] == "Yes").astype(float)
    df["Subsidy_Binary"] = (df["Subsidy_Available"] == "Yes").astype(float)
    df["City_Density_Ordinal"] = df["City_Type"].map(CITY_DENSITY_MAP).astype(float)

    # --- transformations non-linéaires ---
    df["log_income"] = np.log1p(df["Annual_Income_USD"])
    df["sqrt_income"] = np.sqrt(df["Annual_Income_USD"])
    df["sqrt_commute"] = np.sqrt(df["Daily_Commute_km"])
    df["commute_sq"] = df["Daily_Commute_km"] ** 2
    df["age_sq"] = df["Age"].astype(float) ** 2
    df["log_income_sq"] = df["log_income"] ** 2
    df["env_sq"] = df["Environmental_Concern_Level"] ** 2

    # --- ratios revenu / mobilité ---
    df["income_per_car"] = df["Annual_Income_USD"] / df["Number_of_Cars_Owned"]
    df["income_per_commute"] = df["Annual_Income_USD"] / (df["Daily_Commute_km"] + 1.0)
    df["cars_per_income"] = df["Number_of_Cars_Owned"] / (df["Annual_Income_USD"] + 1.0)
    df["commute_per_car"] = df["Daily_Commute_km"] / df["Number_of_Cars_Owned"]
    df["commute_x_cars"] = df["Daily_Commute_km"] * df["Number_of_Cars_Owned"]
    df["age_x_commute"] = df["Age"] * df["Daily_Commute_km"]
    df["age_x_income"] = df["Age"] * df["log_income"]

    # --- infrastructure de recharge ---
    df["Total_Charging_Stations"] = df["Charging_Stations_Near_Home"] + df["Charging_Stations_Near_Work"]
    df["home_work_station_diff"] = df["Charging_Stations_Near_Home"] - df["Charging_Stations_Near_Work"]
    df["home_work_station_ratio"] = df["Charging_Stations_Near_Home"] / (df["Charging_Stations_Near_Work"] + 1.0)
    df["charging_density_relative_to_commute"] = df["Total_Charging_Stations"] / (df["Daily_Commute_km"] + 1.0)
    df["charging_home_relative_to_commute"] = df["Charging_Stations_Near_Home"] / (df["Daily_Commute_km"] + 1.0)
    df["charging_work_relative_to_commute"] = df["Charging_Stations_Near_Work"] / (df["Daily_Commute_km"] + 1.0)
    df["Charging_Access_Index"] = df["Total_Charging_Stations"] * (1.0 + df["Home_Charging_Binary"])
    df["Home_Work_Charging_Index"] = df["Charging_Stations_Near_Home"] * df["Charging_Stations_Near_Work"]

    # --- interactions comportementales (le coeur du signal d'après l'audit) ---
    df["env_x_range_anxiety"] = df["Environmental_Concern_Level"] * df["Range_Anxiety_Ordinal"]
    df["subsidy_x_env"] = df["Subsidy_Binary"] * df["Environmental_Concern_Level"]
    df["home_x_range_anxiety"] = df["Home_Charging_Binary"] * df["Range_Anxiety_Ordinal"]
    df["subsidy_x_home"] = df["Subsidy_Binary"] * df["Home_Charging_Binary"]
    df["subsidy_x_range_anxiety"] = df["Subsidy_Binary"] * df["Range_Anxiety_Ordinal"]
    df["env_x_subsidy_x_home"] = df["Environmental_Concern_Level"] * df["Subsidy_Binary"] * df["Home_Charging_Binary"]
    df["city_density_x_charging"] = df["City_Density_Ordinal"] * df["Total_Charging_Stations"]
    df["city_density_x_commute"] = df["City_Density_Ordinal"] * df["Daily_Commute_km"]
    df["income_x_env"] = df["log_income"] * df["Environmental_Concern_Level"]
    df["income_x_subsidy"] = df["log_income"] * df["Subsidy_Binary"]

    # --- indice "readiness" fait main (résume les leviers dominants identifiés dans l'audit) ---
    df["ev_readiness_score"] = (
        df["Subsidy_Binary"] * 3.0
        + df["Home_Charging_Binary"] * 2.0
        + df["Environmental_Concern_Level"]
        - df["Range_Anxiety_Ordinal"] * 3.0
    )

    return df


def add_rank_features(train: pd.DataFrame, test: pd.DataFrame, cols):
    """Ajoute des rangs percentiles (X-only, calculés sur train+test concaténés -> pas de fuite de cible)."""
    train = train.copy()
    test = test.copy()
    combined = pd.concat([train[cols], test[cols]], axis=0, ignore_index=True)
    ranks = combined.rank(pct=True)
    n_train = len(train)
    for c in cols:
        train[f"{c}_rank"] = ranks[c].to_numpy()[:n_train]
        test[f"{c}_rank"] = ranks[c].to_numpy()[n_train:]
    return train, test


def target_encode_oof(train: pd.DataFrame, test: pd.DataFrame, cols, target_col: str, folds, smoothing: float = 20.0, prefix: str = "te_"):
    """
    Encodage cible strictement out-of-fold pour train, et fit-complet (sur tout train) pour test.
    cols: liste de noms de colonnes (str) ou tuples de colonnes (combinaisons).
    folds: liste de (train_idx, val_idx) -- positions entières dans train (0..len(train)-1).
    """
    train = train.copy()
    test = test.copy()
    y_num = train[target_col]
    global_mean = y_num.mean()

    for col in cols:
        keycols = [col] if isinstance(col, str) else list(col)
        colname = col if isinstance(col, str) else "_".join(col)
        new_col = f"{prefix}{colname}"

        oof_vals = np.full(len(train), np.nan)
        for tr_idx, val_idx in folds:
            tr_fold = train.iloc[tr_idx]
            stats = tr_fold.groupby(keycols, observed=True)[target_col].agg(["mean", "count"]).reset_index()
            stats["smoothed"] = (stats["mean"] * stats["count"] + global_mean * smoothing) / (stats["count"] + smoothing)
            val_keys = train.iloc[val_idx][keycols].reset_index(drop=True)
            merged = val_keys.merge(stats[keycols + ["smoothed"]], on=keycols, how="left")
            oof_vals[val_idx] = merged["smoothed"].fillna(global_mean).to_numpy()
        train[new_col] = oof_vals

        stats_full = train.groupby(keycols, observed=True)[target_col].agg(["mean", "count"]).reset_index()
        stats_full["smoothed"] = (stats_full["mean"] * stats_full["count"] + global_mean * smoothing) / (stats_full["count"] + smoothing)
        test_keys = test[keycols].reset_index(drop=True)
        merged_test = test_keys.merge(stats_full[keycols + ["smoothed"]], on=keycols, how="left")
        test[new_col] = merged_test["smoothed"].fillna(global_mean).to_numpy()

    return train, test


ENGINEERED_COLS = [
    "Range_Anxiety_Ordinal", "Home_Charging_Binary", "Subsidy_Binary", "City_Density_Ordinal",
    "log_income", "sqrt_income", "sqrt_commute", "commute_sq", "age_sq", "log_income_sq", "env_sq",
    "income_per_car", "income_per_commute", "cars_per_income", "commute_per_car", "commute_x_cars",
    "age_x_commute", "age_x_income",
    "Total_Charging_Stations", "home_work_station_diff", "home_work_station_ratio",
    "charging_density_relative_to_commute", "charging_home_relative_to_commute", "charging_work_relative_to_commute",
    "Charging_Access_Index", "Home_Work_Charging_Index",
    "env_x_range_anxiety", "subsidy_x_env", "home_x_range_anxiety", "subsidy_x_home", "subsidy_x_range_anxiety",
    "env_x_subsidy_x_home", "city_density_x_charging", "city_density_x_commute", "income_x_env", "income_x_subsidy",
    "ev_readiness_score",
]

