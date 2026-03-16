"""
pages/page_replay.py -- Race Replay Integration Page

Phase 1: Launches Tom's f1-race-replay engine (MIT license) as a subprocess
         for any completed 2026 race directly from our Streamlit dashboard.

Phase 2 (after Japan R3): Will add prediction overlay showing our model's
         predicted finishing order vs actual outcome side by side.

Credits: Race replay engine from IAmTomShaw/f1-race-replay (MIT License)
         https://github.com/IAmTomShaw/f1-race-replay
"""

import streamlit as st
import subprocess
import sys
import os
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import CALENDAR_2026, RESULTS_2026, TEAM_COLOURS, DRIVERS_2026, POINTS

st.title("Race Replay + Prediction Comparison")
st.caption(
    "Watch any completed 2026 race replay with our ML prediction overlay. "
    "Replay engine adapted from "
    "[IAmTomShaw/f1-race-replay](https://github.com/IAmTomShaw/f1-race-replay) (MIT License)."
)
st.divider()

# ── Find replay_engine path ───────────────────────────────────────────────
PROJECT_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPLAY_ENGINE  = os.path.join(PROJECT_ROOT, "replay_engine")
REPLAY_MAIN    = os.path.join(REPLAY_ENGINE, "main.py")

engine_found = os.path.exists(REPLAY_MAIN)

if not engine_found:
    st.error(
        "replay_engine not found. Run this from your F1_2026 folder:\n\n"
        "```\ngit clone https://github.com/IAmTomShaw/f1-race-replay replay_engine\n```"
    )
    st.stop()

# ── Get completed rounds ──────────────────────────────────────────────────
if "f1_data" in st.session_state and not st.session_state.get("real_2026_df", pd.DataFrame()).empty:
    real = st.session_state["real_2026_df"]
    completed_rounds = sorted(real[real["Year"] == 2026]["Round"].unique().tolist())
else:
    completed_rounds = sorted(RESULTS_2026.keys())

if not completed_rounds:
    st.warning("No completed 2026 races yet. Load data on the Home page first.")
    st.stop()

# ── Race selector ─────────────────────────────────────────────────────────
col1, col2 = st.columns([2, 1])

with col1:
    selected_round = st.selectbox(
        "Select completed race to replay",
        options=completed_rounds,
        format_func=lambda r: f"R{r} — {CALENDAR_2026[r][0]} GP ({CALENDAR_2026[r][2]})",
    )
    race_info = CALENDAR_2026[selected_round]

with col2:
    session_type = st.radio(
        "Session",
        ["Race", "Qualifying"],
        horizontal=True,
    )

st.divider()

# ── Prediction vs Actual comparison ──────────────────────────────────────
st.subheader(f"Prediction vs Actual — {race_info[0]} GP")

col_pred, col_actual = st.columns(2)

with col_actual:
    st.markdown("**Actual Result**")
    if selected_round in RESULTS_2026:
        actual = RESULTS_2026[selected_round]
        actual_rows = []
        for drv, pos in sorted(actual.items(), key=lambda x: x[1]):
            if pos <= 20:
                info = DRIVERS_2026.get(drv, (drv, "Unknown", ""))
                pts  = POINTS.get(pos, 0)
                actual_rows.append({
                    "Pos":    pos,
                    "Driver": drv,
                    "Name":   info[0].split()[0] + " " + info[0].split()[-1],
                    "Team":   info[1],
                    "Points": pts,
                })
        actual_df = pd.DataFrame(actual_rows).sort_values("Pos")
        st.dataframe(actual_df, use_container_width=True, hide_index=True, height=420)
    else:
        st.info("No result data for this round yet.")

with col_pred:
    st.markdown("**Our Model's Pre-Race Prediction**")
    if st.session_state.get("model_ready") and st.session_state.get("model_obj"):
        model   = st.session_state["model_obj"]
        rolling = st.session_state["rolling_df"]

        if rolling is not None:
            from models.predictor import predict_race

            latest = (
                rolling[rolling["Round"] < selected_round]
                .sort_values(["Driver", "Year", "Round"])
                .groupby("Driver").last()
                .reset_index()
            )

            # Fill missing drivers
            from models.predictor import DRIVER_EXP
            known = set(latest["Driver"].unique())
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
                latest = pd.concat([latest, pd.DataFrame(filler)], ignore_index=True)

            prediction = predict_race(model, latest)
            pred_display = prediction[["PredictedPos", "Driver", "Name", "Team",
                                       "PredictedPoints", "Confidence"]].copy()
            pred_display.columns = ["Pos", "Code", "Driver", "Team", "Pts", "Confidence"]
            st.dataframe(pred_display, use_container_width=True,
                         hide_index=True, height=420)
        else:
            st.info("Train the model on the Next Race Predictor page first.")
    else:
        st.info("Train the model on the **Next Race Predictor** page to see predictions here.")

# ── Accuracy score for completed races ───────────────────────────────────
if (selected_round in RESULTS_2026 and
        st.session_state.get("model_ready") and
        st.session_state.get("rolling_df") is not None):

    st.divider()
    st.subheader("Prediction Accuracy")

    actual_dict = {drv: pos for drv, pos in RESULTS_2026[selected_round].items() if pos <= 20}

    if "prediction" in dir() and not prediction.empty:
        pred_dict = dict(zip(prediction["Driver"], prediction["PredictedPos"]))

        common_drivers = set(actual_dict.keys()) & set(pred_dict.keys())
        if common_drivers:
            errors = [abs(pred_dict[d] - actual_dict[d]) for d in common_drivers]
            mae    = sum(errors) / len(errors)

            # Top 3 accuracy
            actual_top3 = {d for d, p in actual_dict.items() if p <= 3}
            pred_top3   = {d for d, p in pred_dict.items()   if p <= 3}
            podium_hits = len(actual_top3 & pred_top3)

            # Top 10 accuracy
            actual_top10 = {d for d, p in actual_dict.items() if p <= 10}
            pred_top10   = {d for d, p in pred_dict.items()   if p <= 10}
            top10_hits   = len(actual_top10 & pred_top10)

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("MAE (positions)",    f"{mae:.2f}")
            k2.metric("Podium drivers hit", f"{podium_hits}/3")
            k3.metric("Top 10 drivers hit", f"{top10_hits}/10")
            k4.metric("Drivers compared",   len(common_drivers))

            # Driver-by-driver delta
            delta_rows = []
            for drv in sorted(common_drivers, key=lambda d: actual_dict[d]):
                info = DRIVERS_2026.get(drv, (drv, "Unknown", ""))
                delta_rows.append({
                    "Driver":        drv,
                    "Name":          info[0].split()[0] + " " + info[0].split()[-1],
                    "Team":          info[1],
                    "Actual Pos":    actual_dict[drv],
                    "Predicted Pos": pred_dict.get(drv, "-"),
                    "Error":         abs(pred_dict.get(drv, 20) - actual_dict[drv]),
                })

            delta_df = pd.DataFrame(delta_rows).sort_values("Actual Pos")

            import plotly.express as px
            fig = px.bar(
                delta_df,
                x="Name",
                y="Error",
                color="Team",
                color_discrete_map=TEAM_COLOURS,
                title=f"Prediction error per driver — {race_info[0]} GP",
                labels={"Error": "Position error (|predicted - actual|)", "Name": "Driver"},
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                showlegend=False,
                height=350,
                xaxis=dict(tickangle=-45),
            )
            st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── Launch replay button ──────────────────────────────────────────────────
st.subheader(f"Launch Race Replay — {race_info[0]} GP")
st.caption(
    "Opens Tom's race replay engine in a separate window. "
    "Cars animate as dots around the circuit with live leaderboard."
)

col_btn, col_info = st.columns([1, 2])

with col_btn:
    flag = "--qualifying" if session_type == "Qualifying" else ""

    if st.button(f"Launch {session_type} Replay", type="primary", use_container_width=True):
        cmd = [sys.executable, REPLAY_MAIN,
               "--viewer",
               "--year", "2026",
               "--round", str(selected_round)]
        if flag:
            cmd.append(flag)

        try:
            subprocess.Popen(
                cmd,
                cwd=REPLAY_ENGINE,
                creationflags=subprocess.CREATE_NEW_CONSOLE
                              if sys.platform == "win32" else 0,
            )
            st.success(
                f"Launched! R{selected_round} {race_info[0]} GP replay opening in a new window. "
                "First launch downloads data (~1 min), subsequent launches are instant."
            )
        except Exception as e:
            st.error(f"Failed to launch replay: {e}")

with col_info:
    st.markdown("""
    **Controls once replay opens:**
    - `SPACE` — Pause / Resume
    - `Arrow keys` — Speed up / slow down
    - `R` — Restart from lap 1
    - `1-4` — Set speed (0.5x / 1x / 2x / 4x)
    - `Click driver` — Show telemetry (speed, gear, DRS)
    - `D` — Toggle DRS zones
    """)