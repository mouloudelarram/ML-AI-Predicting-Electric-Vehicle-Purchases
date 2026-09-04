"""Configuration partagée pour le pipeline ML - Predicting EV Purchases."""
import os

SEED = 42
N_FOLDS = 5

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_PATH = os.path.join(ROOT, "train.csv")
TEST_PATH = os.path.join(ROOT, "test.csv")

OUT_DIR = os.path.join(ROOT, "outputs")
OOF_DIR = os.path.join(OUT_DIR, "oof")
PRED_DIR = os.path.join(OUT_DIR, "preds")
SUB_DIR = os.path.join(OUT_DIR, "submissions")
LOG_DIR = os.path.join(OUT_DIR, "logs")
MODEL_DIR = os.path.join(OUT_DIR, "models")

for d in [OUT_DIR, OOF_DIR, PRED_DIR, SUB_DIR, LOG_DIR, MODEL_DIR]:
    os.makedirs(d, exist_ok=True)

ID_COL = "id"
TARGET_COL = "Will_Buy_EV"

CAT_COLS = [
    "Gender",
    "City_Type",
    "Current_Car_Type",
    "Home_Charging_Possible",
    "Subsidy_Available",
    "Range_Anxiety_Level",
]

NUM_COLS = [
    "Age",
    "Annual_Income_USD",
    "Daily_Commute_km",
    "Number_of_Cars_Owned",
    "Charging_Stations_Near_Home",
    "Charging_Stations_Near_Work",
    "Environmental_Concern_Level",
]

BASE_FEATURES = NUM_COLS + CAT_COLS

