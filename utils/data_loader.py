"""
utils/data_loader.py -- ETL pipeline for F1 2026 Prediction System.

Data priority order:
  1. FastF1 v3.8.1 -- REAL 2026 data (Australia, China already available)
  2. FastF1 -- 2023, 2024 historical training data
  3. Jolpica API -- fallback if FastF1 session fails
  4. config.RESULTS_2026 -- hardcoded last resort fallback

FastF1 v3.8.1 (Feb 2026) natively supports 2026 season.
Run: pip install --upgrade fastf1
"""

import warnings
import requests
import fastf1
import pandas as pd
import numpy as np
warnings.filterwarnings("ignore")

from config import (
    CACHE_DIR, TRAIN_SEASONS, RESULTS_2026,
    DRIVERS_2026, CALENDAR_2026, POINTS, ERA_WEIGHTS
)

fastf1.Cache.enable_cache(CACHE_DIR)


def load_session(year: int, round_number: int, session_type: str = "R"):
    """
    Load any race session via FastF1. Works for 2023, 2024, and 2026.
    Explicitly calls session.load() with laps=True to ensure lap data
    is available before returning. Returns None on any failure.
    """
    try:
        session = fastf1.get_session(year, round_number, session_type)
        # laps=True is required -- without it session.laps raises DataNotLoadedError
        session.load(laps=True, telemetry=False, weather=False, messages=False)
        print(f"  Loaded {year} R{round_number} ({session_type}): {session.event['EventName']}")
        return session
    except Exception as e:
        print(f"  FastF1 load failed {year} R{round_number}: {e}")
        return None


def extract_laps(session) -> pd.DataFrame:
    """
    Clean and enrich lap data from a FastF1 session.
    Guards against DataNotLoadedError -- returns empty DataFrame if laps
    were not loaded rather than crashing.
    """
    if session is None:
        return pd.DataFrame()

    # Guard: check laps are actually loaded before accessing
    try:
        laps = session.laps.copy()
    except Exception as e:
        print(f"  Laps not available ({e}) -- skipping lap features for this session")
        return pd.DataFrame()

    if laps.empty:
        return pd.DataFrame()

    laps = laps.dropna(subset=["LapTime"])
    if laps.empty:
        return pd.DataFrame()

    laps["LapTimeSec"] = laps["LapTime"].dt.total_seconds()

    # Remove SC/pit laps (>15% slower than median)
    median_time = laps["LapTimeSec"].median()
    if pd.isna(median_time) or median_time <= 0:
        return pd.DataFrame()
    laps = laps[laps["LapTimeSec"] < median_time * 1.15]

    laps["Circuit"] = session.event["EventName"]
    laps["Round"]   = session.event["RoundNumber"]
    laps["Year"]    = session.event.year

    # For 2026: map car numbers to driver abbreviations
    # FastF1 uses DriverNumber internally; "Driver" column has the number as string
    if session.event.year == 2026 and "Driver" in laps.columns:
        from config import CAR_NUMBERS_2026
        laps["Driver"] = laps["Driver"].apply(
            lambda d: CAR_NUMBERS_2026.get(int(d), d) if str(d).isdigit() else d
        )

    # Pace delta vs field median
    med = laps.groupby("Round")["LapTimeSec"].median()
    laps["MedianTime"] = laps["Round"].map(med)
    laps["PaceDelta"]  = laps["LapTimeSec"] - laps["MedianTime"]

    return laps.reset_index(drop=True)


def extract_results(session) -> pd.DataFrame:
    """
    Clean race results from a FastF1 session.
    Guards against DataNotLoadedError on results access.
    """
    if session is None:
        return pd.DataFrame()

    try:
        res = session.results.copy()
    except Exception as e:
        print(f"  Results not available ({e}) -- skipping this session")
        return pd.DataFrame()

    if res.empty:
        return pd.DataFrame()

    res["Round"]   = session.event["RoundNumber"]
    res["Year"]    = session.event.year
    res["Circuit"] = session.event["EventName"]
    for col in ["Points", "GridPosition", "Position"]:
        if col in res.columns:
            res[col] = pd.to_numeric(res[col], errors="coerce")
    return res.reset_index(drop=True)


def build_feature_rows(results, laps, year, round_number):
    """
    Convert raw results + laps into ML-ready feature rows.
    Handles both abbreviation-based and car-number-based driver identification.
    For 2026: car numbers changed (Verstappen #3, Norris #1 etc) --
    uses CAR_NUMBERS_2026 map to resolve correctly.
    """
    from config import CAR_NUMBERS_2026
    rows = []
    for _, dr in results.iterrows():
        # Try abbreviation first
        code = str(dr.get("Abbreviation", "")).strip()

        # If abbreviation missing or looks like a number, resolve via car number
        if not code or code.isdigit():
            car_num = dr.get("DriverNumber", None)
            if car_num is not None:
                try:
                    code = CAR_NUMBERS_2026.get(int(car_num), str(car_num))
                except (ValueError, TypeError):
                    code = str(car_num)

        if not code:
            continue
        drv_laps   = laps[laps["Driver"] == code] if not laps.empty else pd.DataFrame()
        finish_pos = pd.to_numeric(dr.get("Position",    20), errors="coerce")
        grid_pos   = pd.to_numeric(dr.get("GridPosition", 20), errors="coerce")
        points     = pd.to_numeric(dr.get("Points",        0), errors="coerce")
        team       = str(dr.get("TeamName", "Unknown"))
        med_pace   = float(drv_laps["PaceDelta"].median()) if not drv_laps.empty and "PaceDelta" in drv_laps.columns else 0.0
        pace_std   = float(drv_laps["PaceDelta"].std())    if not drv_laps.empty and len(drv_laps) > 2 else 1.0
        rows.append({
            "Year":            year,
            "Round":           round_number,
            "Driver":          code,
            "Team":            team,
            "GridPosition":    float(grid_pos)   if not pd.isna(grid_pos)   else 15.0,
            "FinishPos":       float(finish_pos) if not pd.isna(finish_pos) else 20.0,
            "Points":          float(points)     if not pd.isna(points)     else 0.0,
            "MedianPaceDelta": med_pace,
            "PaceConsistency": pace_std,
            "EraWeight":       ERA_WEIGHTS.get(year, 1.0),
        })
    return pd.DataFrame(rows)


def load_2026_real_data() -> pd.DataFrame:
    """
    PRIMARY SOURCE: Load actual 2026 race data from FastF1 v3.8.1.
    Australia R1 and China R2 are already available.
    Japan R3 (Mar 29) available after the race.
    Each new race auto-loads as the season progresses.
    """
    print("Loading 2026 real data via FastF1 v3.8.1...")
    all_rows = []
    for rnd in range(1, 23):
        session = load_session(2026, rnd, "R")
        if session is None:
            break   # race hasn't happened yet
        laps    = extract_laps(session)
        results = extract_results(session)
        if results.empty:
            break
        rows = build_feature_rows(results, laps, 2026, rnd)
        if rows is not None and not rows.empty:
            all_rows.append(rows)

    if all_rows:
        df = pd.concat(all_rows, ignore_index=True)
        print(f"  2026 FastF1 real data: {len(df)} rows, {df['Round'].nunique()} races")
        return df

    print("  FastF1 2026 unavailable -- using hardcoded config fallback")
    return build_2026_features_from_config()


def build_2026_features_from_config() -> pd.DataFrame:
    """Fallback: hardcoded R1/R2 results from config.py."""
    rows = []
    for rnd, driver_positions in RESULTS_2026.items():
        for drv_code, finish_pos in driver_positions.items():
            info = DRIVERS_2026.get(drv_code, (drv_code, "Unknown", "Unknown"))
            rows.append({
                "Year":            2026,
                "Round":           rnd,
                "Driver":          drv_code,
                "Team":            info[1],
                "GridPosition":    float(finish_pos),
                "FinishPos":       float(min(finish_pos, 20)),
                "Points":          float(POINTS.get(min(finish_pos, 20), 0)),
                "MedianPaceDelta": 0.0,
                "PaceConsistency": 1.0,
                "EraWeight":       ERA_WEIGHTS.get(2026, 2.5),
            })
    return pd.DataFrame(rows)


def load_2026_testing_data() -> pd.DataFrame:
    """Load 2026 pre-season testing via fastf1.get_testing_session(2026, 1, n)."""
    print("Loading 2026 pre-season testing data...")
    rows = []
    for session_num in [1, 2, 3]:
        try:
            session = fastf1.get_testing_session(2026, 1, session_num)
            session.load(laps=True, telemetry=False, weather=False, messages=False)
            laps = extract_laps(session)
            if laps.empty:
                continue
            best = laps.sort_values("LapTimeSec").groupby("Driver").first().reset_index()
            for _, row in best.iterrows():
                drv  = row["Driver"]
                info = DRIVERS_2026.get(drv, (drv, "Unknown", "Unknown"))
                rows.append({
                    "Year": 2026, "Round": 0, "Driver": drv, "Team": info[1],
                    "GridPosition": 10.0, "FinishPos": 10.0, "Points": 0.0,
                    "MedianPaceDelta": float(row.get("PaceDelta", 0)),
                    "PaceConsistency": 1.0, "EraWeight": 1.5,
                })
            print(f"  Testing session {session_num}: {len(rows)} driver rows")
        except Exception as e:
            print(f"  Testing session {session_num} not available: {e}")
            break
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def load_training_data(seasons=TRAIN_SEASONS, max_rounds_per_season=22, progress_callback=None):
    """Load 2023 + 2024 historical data for model training."""
    all_rows = []
    for year in seasons:
        try:
            schedule = fastf1.get_event_schedule(year, include_testing=False)
            n_rounds = min(len(schedule), max_rounds_per_season)
        except Exception:
            n_rounds = max_rounds_per_season

        for rnd in range(1, n_rounds + 1):
            if progress_callback:
                progress_callback(f"Loading {year} R{rnd}/{n_rounds}...")
            session = load_session(year, rnd, "R")
            if session is None:
                continue
            laps    = extract_laps(session)
            results = extract_results(session)
            if results.empty:
                continue
            rows = build_feature_rows(results, laps, year, rnd)
            if rows is not None and not rows.empty:
                all_rows.append(rows)

    if not all_rows:
        return pd.DataFrame()
    df = pd.concat(all_rows, ignore_index=True)
    print(f"Training data: {len(df)} rows, {df['Year'].nunique()} seasons")
    return df


def build_full_dataset(max_historical_rounds=22, progress_callback=None):
    """
    Build complete dataset: 2023 + 2024 history + real 2026 data.
    Called once on app startup; cached in st.session_state.
    """
    print("=== Building full dataset ===")
    hist      = load_training_data(max_rounds_per_season=max_historical_rounds, progress_callback=progress_callback)
    real_2026 = load_2026_real_data()
    test_2026 = load_2026_testing_data()

    frames = [f for f in [hist, real_2026, test_2026] if not f.empty]
    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    print(f"=== Dataset ready: {len(combined)} total rows ===")
    print(f"    2023: {len(combined[combined['Year']==2023])} rows")
    print(f"    2024: {len(combined[combined['Year']==2024])} rows")
    print(f"    2026: {len(combined[combined['Year']==2026])} rows")
    return combined


def compute_rolling_stats(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Rolling averages per driver: AvgFinish, AvgGrid, RecentForm, DNFRate.
    Safely handles missing columns by filling with sensible defaults.
    """
    if df.empty:
        return df

    # Ensure required columns exist with defaults
    if "Points" not in df.columns:
        df = df.copy()
        df["Points"] = 0.0
    if "FinishPos" not in df.columns:
        df = df.copy()
        df["FinishPos"] = 15.0
    if "GridPosition" not in df.columns:
        df = df.copy()
        df["GridPosition"] = 15.0

    df = df.sort_values(["Driver", "Year", "Round"]).copy()
    parts = []
    for drv, grp in df.groupby("Driver"):
        grp = grp.copy()
        grp["AvgFinish"]  = grp["FinishPos"].rolling(window, min_periods=1).mean()
        grp["AvgGrid"]    = grp["GridPosition"].rolling(window, min_periods=1).mean()
        grp["RecentForm"] = grp["Points"].rolling(window, min_periods=1).mean()
        grp["DNFRate"]    = (grp["FinishPos"] > 15).rolling(window, min_periods=1).mean()
        parts.append(grp)
    return pd.concat(parts).sort_values(["Year", "Round", "Driver"]).reset_index(drop=True)


def compute_standings(results_dict=None, df_2026=None) -> pd.DataFrame:
    """
    Compute championship standings including Sprint points.
    Prefers real FastF1 data over hardcoded when available.
    Sprint points: 8-7-6-5-4-3-2-1 for top 8 finishers.
    """
    from config import SPRINT_RESULTS_2026, SPRINT_POINTS
    standings = {}

    # Use real FastF1 data if available
    if df_2026 is not None and not df_2026.empty:
        real = df_2026[df_2026["Year"] == 2026]
        for _, row in real.iterrows():
            drv = row["Driver"]
            standings[drv] = standings.get(drv, 0) + float(row.get("Points", 0))

    # Use hardcoded race results
    elif results_dict:
        for rnd in sorted(results_dict.keys()):
            for drv, pos in results_dict[rnd].items():
                pts = POINTS.get(pos, 0)
                standings[drv] = standings.get(drv, 0) + pts

    # Always add sprint points on top (not in FastF1 race data)
    for rnd, sprint_result in SPRINT_RESULTS_2026.items():
        for drv, pos in sprint_result.items():
            pts = SPRINT_POINTS.get(pos, 0)
            if pts > 0:
                standings[drv] = standings.get(drv, 0) + pts

    rows = []
    for drv_code, pts in standings.items():
        info = DRIVERS_2026.get(drv_code, (drv_code, "Unknown", "Unknown"))
        rows.append({"Driver": drv_code, "Name": info[0], "Team": info[1], "Points": pts})

    df = pd.DataFrame(rows).sort_values("Points", ascending=False).reset_index(drop=True)
    df.index += 1
    df.index.name = "Pos"
    return df


def fetch_jolpica(round_number: int, year: int = 2026) -> dict:
    """Jolpica API fallback when FastF1 fails."""
    url = f"https://api.jolpi.ca/ergast/f1/{year}/{round_number}/results.json"
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            return {}
        races = resp.json()["MRData"]["RaceTable"]["Races"]
        if not races:
            return {}
        return {r["Driver"]["code"]: int(r["position"]) for r in races[0]["Results"]}
    except Exception as e:
        print(f"  Jolpica failed: {e}")
        return {}