"""
pages/page_race_replay.py -- Race Replay Page

Launches replay_window.py as a separate process.
The PySide6 window opens with animated cars on track,
play/pause controls, leaderboard, and driver telemetry panel.
"""

import streamlit as st
import subprocess
import sys
import os
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import CALENDAR_2026, RESULTS_2026, DRIVERS_2026, TEAM_COLOURS, POINTS

PROJECT_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPLAY_SCRIPT  = os.path.join(PROJECT_ROOT, "replay_window.py")

st.title("Race Replay")
st.caption(
    "Animated desktop window with cars moving on track, "
    "live leaderboard, and driver telemetry panel."
)
st.divider()

# ── Session selector ──────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)

with col1:
    year = st.selectbox("Season", [2026, 2024, 2023], index=0)

with col2:
    if year == 2026:
        completed  = sorted(RESULTS_2026.keys())
        round_opts = completed
        fmt = lambda r: f"R{r} -- {CALENDAR_2026.get(r, ('?',))[0]}"
    else:
        round_opts = list(range(1, 24))
        fmt = lambda r: f"Round {r}"

    selected_round = st.selectbox("Round", round_opts, format_func=fmt)

with col3:
    session_type = st.selectbox(
        "Session", ["R", "Q"],
        format_func=lambda s: "Race" if s == "R" else "Qualifying"
    )

st.divider()

# ── Launch button ─────────────────────────────────────────────────────────
circuit = CALENDAR_2026.get(selected_round, (f"Round {selected_round}", "", ""))[0] \
          if year == 2026 else f"Round {selected_round}"

col_btn, col_info = st.columns([1, 2])

with col_btn:
    if st.button(f"Launch Replay Window", type="primary", use_container_width=True):
        if not os.path.exists(REPLAY_SCRIPT):
            st.error("replay_window.py not found in project root.")
        else:
            try:
                cmd = [
                    sys.executable,
                    REPLAY_SCRIPT,
                    "--year",    str(year),
                    "--round",   str(selected_round),
                    "--session", session_type,
                ]
                subprocess.Popen(
                    cmd,
                    cwd=PROJECT_ROOT,
                    creationflags=subprocess.CREATE_NEW_CONSOLE
                                  if sys.platform == "win32" else 0,
                )
                st.success(
                    f"Opening {year} R{selected_round} {circuit} replay window... "
                    f"First load downloads telemetry (~1-2 min). Cached after that."
                )
            except Exception as e:
                st.error(f"Failed to launch: {e}")

with col_info:
    st.markdown(f"""
    **What will open:**
    A desktop window showing the **{circuit} {year}** race.

    **Controls in the window:**
    - `Play / Pause` button
    - Speed selector: 0.25x / 0.5x / 1x / 2x / 4x
    - Scrubber bar to jump to any point in the race
    - `Restart` to go back to lap 1
    - Click any car dot to see speed, gear, DRS, tyre

    **Track colours:**
    - Dark gray = normal racing
    - Orange = Safety Car deployed
    - Yellow = VSC / Yellow flag
    - Red = Red flag
    """)

st.divider()

# ── Prediction vs Actual comparison ──────────────────────────────────────
if year == 2026 and selected_round in RESULTS_2026:
    st.subheader(f"Prediction vs Actual -- {circuit} GP")

    col_pred, col_actual = st.columns(2)

    with col_actual:
        st.markdown("**Actual Result**")
        actual = RESULTS_2026[selected_round]
        rows   = []
        for drv, pos in sorted(actual.items(), key=lambda x: x[1]):
            if pos <= 20:
                info = DRIVERS_2026.get(drv, (drv, "Unknown", ""))
                rows.append({
                    "Pos":    pos,
                    "Driver": drv,
                    "Name":   info[0].split()[0] + " " + info[0].split()[-1],
                    "Team":   info[1],
                    "Points": POINTS.get(pos, 0),
                })
        st.dataframe(
            pd.DataFrame(rows), use_container_width=True,
            hide_index=True, height=420
        )

    with col_pred:
        st.markdown("**Our Model's Prediction**")
        if (st.session_state.get("model_ready") and
                st.session_state.get("rolling_df") is not None):

            from models.predictor import predict_race

            rolling = st.session_state["rolling_df"]
            model   = st.session_state["model_obj"]

            latest = (
                rolling[rolling["Round"] < selected_round]
                .sort_values(["Driver", "Year", "Round"])
                .groupby("Driver").last()
                .reset_index()
            )

            known  = set(latest["Driver"].unique())
            filler = []
            for drv in DRIVERS_2026:
                if drv not in known:
                    info = DRIVERS_2026[drv]
                    filler.append({
                        "Driver": drv, "Team": info[1],
                        "GridPosition": 15.0, "FinishPos": 15.0,
                        "AvgFinish": 15.0, "AvgGrid": 15.0,
                        "RecentForm": 2.0, "DNFRate": 0.1,
                        "MedianPaceDelta": 0.0, "PaceConsistency": 1.5,
                        "EraWeight": 2.5,
                    })
            if filler:
                latest = pd.concat(
                    [latest, pd.DataFrame(filler)], ignore_index=True)

            pred = predict_race(model, latest)
            disp = pred[["PredictedPos", "Driver", "Name", "Team",
                          "PredictedPoints", "Confidence"]].copy()
            disp.columns = ["Pos", "Code", "Driver", "Team",
                             "Pts", "Confidence"]
            st.dataframe(
                disp, use_container_width=True,
                hide_index=True, height=420
            )

            # Accuracy metrics
            actual_dict = {d: p for d, p in actual.items() if p <= 20}
            pred_dict   = dict(zip(pred["Driver"], pred["PredictedPos"]))
            common      = set(actual_dict) & set(pred_dict)

            if common:
                mae = sum(abs(pred_dict[d] - actual_dict[d])
                          for d in common) / len(common)
                podium_hits = len(
                    {d for d, p in actual_dict.items() if p <= 3} &
                    {d for d, p in pred_dict.items()   if p <= 3}
                )
                top10_hits = len(
                    {d for d, p in actual_dict.items() if p <= 10} &
                    {d for d, p in pred_dict.items()   if p <= 10}
                )
                st.divider()
                k1, k2, k3 = st.columns(3)
                k1.metric("MAE",            f"{mae:.2f} pos")
                k2.metric("Podium correct",  f"{podium_hits}/3")
                k3.metric("Top 10 correct",  f"{top10_hits}/10")
        else:
            st.info(
                "Train the model on **Next Race Predictor** "
                "to see predictions here."
            )