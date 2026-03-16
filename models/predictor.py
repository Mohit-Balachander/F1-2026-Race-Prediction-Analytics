"""
models/predictor.py -- 2026 F1 Race Prediction Model

Architecture:
  - XGBoost regressor predicting finishing position
  - Trained on 2023 + 2024 FastF1 data
  - Up-weighted 2026 results via sample_weight (era weighting)
  - Features: rolling form, qualifying pos, team strength, driver consistency
  - Regulation era weight: 2026 data counts 2.5x more than 2023 data
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings("ignore")

from config import DRIVERS_2026, CALENDAR_2026, CIRCUIT_TYPES, TEAM_COLOURS, POINTS


# ── Feature columns used by the model ─────────────────────────────────────
FEATURE_COLS = [
    "GridPosition",
    "AvgFinish",
    "AvgGrid",
    "RecentForm",
    "DNFRate",
    "MedianPaceDelta",
    "PaceConsistency",
    "TeamStrength",
    "DriverExp",
]
TARGET_COL = "FinishPos"


# ── Team strength scores (0-10, based on 2025 + 2026 R1-R2 performance) ──
# Updated manually: Mercedes clearly dominant in 2026 new regs
TEAM_STRENGTH_2026 = {
    "Mercedes":     9.5,
    "Ferrari":      7.5,
    "McLaren":      7.0,   # 2025 champs but early 2026 disasters
    "Red Bull":     6.5,
    "Aston Martin": 6.0,   # Newey factor
    "Williams":     5.5,
    "Alpine":       5.0,
    "Racing Bulls": 5.0,
    "Haas":         4.5,
    "Audi":         4.0,
    "Cadillac":     3.5,   # brand new team
}

# Driver experience scores (races in F1 up to 2026 start)
DRIVER_EXP = {
    "RUS": 120, "ANT": 20,  "LEC": 160, "HAM": 350,
    "NOR": 130, "PIA": 60,  "VER": 200, "HAD": 20,
    "ALO": 380, "STR": 170, "ALB": 100, "SAI": 200,
    "GAS": 180, "COL": 10,  "LAW": 30,  "LIN": 1,
    "BEA": 30,  "OCO": 160, "HUL": 200, "BOR": 20,
    "PER": 260, "BOT": 220,
}


# ── 1. Prepare features ────────────────────────────────────────────────────
def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add team strength and driver experience to the feature DataFrame.
    Map team names to strength scores.
    """
    df = df.copy()

    # Team strength -- try 2026 mapping first, fall back to generic
    def team_strength(team_name):
        for key in TEAM_STRENGTH_2026:
            if key.lower() in str(team_name).lower():
                return TEAM_STRENGTH_2026[key]
        return 5.0

    df["TeamStrength"]  = df["Team"].apply(team_strength)
    df["DriverExp"]     = df["Driver"].map(DRIVER_EXP).fillna(50)

    # Fill any remaining NaNs
    for col in FEATURE_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median() if df[col].notna().any() else 10.0)

    return df


# ── 2. Train the model ─────────────────────────────────────────────────────
def train_model(df: pd.DataFrame):
    """
    Train an ensemble of GradientBoosting + RandomForest on the feature DataFrame.
    Returns the fitted model and a performance summary dict.
    """
    df = prepare_features(df)

    # Keep only rows with all required features
    available = [c for c in FEATURE_COLS if c in df.columns]
    df_model  = df.dropna(subset=available + [TARGET_COL]).copy()

    if len(df_model) < 30:
        print("  WARNING: Very few training samples. Load more historical races.")

    X = df_model[available].values
    y = df_model[TARGET_COL].values
    w = df_model["EraWeight"].values if "EraWeight" in df_model.columns \
        else np.ones(len(df_model))

    # ── Primary model: Gradient Boosting ──────────────────────────────────
    gb = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        random_state=42,
    )
    gb.fit(X, y, sample_weight=w)

    # ── Secondary model: Random Forest (for ensemble) ─────────────────────
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=6,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X, y, sample_weight=w)

    # ── Cross-validation MAE ───────────────────────────────────────────────
    cv_scores = cross_val_score(gb, X, y, cv=5, scoring="neg_mean_absolute_error")
    mae       = -cv_scores.mean()

    # ── Feature importance ─────────────────────────────────────────────────
    importance = pd.DataFrame({
        "Feature":    available,
        "Importance": gb.feature_importances_,
    }).sort_values("Importance", ascending=False)

    model_obj = {
        "gb":         gb,
        "rf":         rf,
        "features":   available,
        "mae":        round(mae, 3),
        "n_samples":  len(df_model),
        "importance": importance,
    }

    print(f"  Model trained. MAE = {mae:.3f} positions | n = {len(df_model)}")
    return model_obj


# ── 3. Predict one race ────────────────────────────────────────────────────
def predict_race(
    model_obj: dict,
    driver_features: pd.DataFrame,
    qualifying_positions: dict = None,
) -> pd.DataFrame:
    """
    Predict finishing order for one upcoming race.
    Only returns predictions for drivers on the 2026 grid.
    Old drivers from training data (ZHO, RIC, MAG etc) are filtered out.
    """
    df = prepare_features(driver_features.copy())

    # CRITICAL: filter to 2026 grid only -- removes old drivers from training data
    df = df[df["Driver"].isin(DRIVERS_2026.keys())].copy()

    if df.empty:
        return pd.DataFrame()

    # If qualifying positions provided, use them as GridPosition
    if qualifying_positions:
        df["GridPosition"] = df["Driver"].map(qualifying_positions).fillna(df["GridPosition"])

    available = model_obj["features"]
    for col in available:
        if col not in df.columns:
            df[col] = 10.0
        df[col] = df[col].fillna(10.0)

    X = df[available].values

    gb = model_obj["gb"]
    rf = model_obj["rf"]

    pred_gb = gb.predict(X)
    pred_rf = rf.predict(X)
    pred    = 0.6 * pred_gb + 0.4 * pred_rf

    df["PredictedPos_Raw"] = pred

    # Re-rank to exactly 1-22
    df = df.sort_values("PredictedPos_Raw").reset_index(drop=True)
    df["PredictedPos"] = range(1, len(df) + 1)

    df["ModelDisagreement"] = abs(pred_gb - pred_rf)
    df["Confidence"] = df["ModelDisagreement"].apply(
        lambda x: "High" if x < 1.5 else ("Medium" if x < 3.0 else "Low")
    )

    df["Name"] = df["Driver"].map(lambda d: DRIVERS_2026.get(d, (d, "", ""))[0])
    df["Team"] = df["Driver"].map(lambda d: DRIVERS_2026.get(d, (d, "Unknown", ""))[1])
    df["PredictedPoints"] = df["PredictedPos"].map(lambda p: POINTS.get(p, 0))

    cols = ["PredictedPos", "Driver", "Name", "Team",
            "PredictedPoints", "Confidence", "PredictedPos_Raw"]
    return df[[c for c in cols if c in df.columns]].reset_index(drop=True)


# ── 4. Forecast full championship ─────────────────────────────────────────
def forecast_championship(
    model_obj: dict,
    driver_rolling: pd.DataFrame,
    known_results: dict,
    remaining_rounds: list,
) -> pd.DataFrame:
    """
    Project final championship standings by predicting all remaining races.

    Parameters
    ----------
    model_obj       : trained model
    driver_rolling  : rolling stats DataFrame
    known_results   : {round: {driver: pos}} -- already completed races
    remaining_rounds: list of round numbers still to race

    Returns
    -------
    DataFrame with projected final points per driver.
    """
    from utils.data_loader import compute_standings

    # Start from known points
    current_standings = compute_standings(known_results)
    projected_points  = dict(zip(
        current_standings["Driver"],
        current_standings["Points"]
    ))

    # Predict each remaining race
    for rnd in remaining_rounds:
        # Use most recent rolling features for each driver
        latest = driver_rolling.sort_values(["Driver", "Year", "Round"])\
                               .groupby("Driver").last().reset_index()

        prediction = predict_race(model_obj, latest)

        for _, row in prediction.iterrows():
            drv = row["Driver"]
            pts = int(row.get("PredictedPoints", 0))
            projected_points[drv] = projected_points.get(drv, 0) + pts

    # Build final standings table
    rows = []
    for drv_code, pts in projected_points.items():
        info = DRIVERS_2026.get(drv_code, (drv_code, "Unknown", "Unknown"))
        rows.append({
            "Driver": drv_code,
            "Name":   info[0],
            "Team":   info[1],
            "ProjectedPoints": pts,
        })

    df = pd.DataFrame(rows)\
           .sort_values("ProjectedPoints", ascending=False)\
           .reset_index(drop=True)
    df.index += 1
    df.index.name = "Projected Pos"
    return df