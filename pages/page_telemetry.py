"""
pages/page_telemetry.py -- Telemetry Analysis Page

Built using FastF1's telemetry data directly.
Inspired by Tom Shaw's f1-race-replay but built independently
using Plotly instead of Arcade for browser-native display.

Features:
  - Driver vs driver lap time comparison
  - Speed / throttle / brake / gear traces vs distance
  - Tyre strategy chart
  - Fastest lap telemetry overlay
  - Qualifying lap comparison
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fastf1
from config import (
    CACHE_DIR, CALENDAR_2026, TEAM_COLOURS,
    DRIVER_COLOURS, DRIVERS_2026, CAR_NUMBERS_2026
)

fastf1.Cache.enable_cache(CACHE_DIR)

st.title("Telemetry Analysis")
st.caption(
    "Driver vs driver speed, throttle, brake and gear traces "
    "from official F1 timing data via FastF1."
)
st.divider()

# ── Session selector ──────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)

with col1:
    year = st.selectbox("Season", [2026, 2024, 2023], index=0)

with col2:
    if year == 2026:
        rounds = list(CALENDAR_2026.keys())
        round_labels = {r: f"R{r} — {CALENDAR_2026[r][0]}" for r in rounds}
    else:
        round_labels = {r: f"Round {r}" for r in range(1, 25)}
        rounds = list(range(1, 25))

    selected_round = st.selectbox(
        "Race",
        options=rounds[:5],
        format_func=lambda r: round_labels.get(r, f"Round {r}"),
    )

with col3:
    session_type = st.selectbox(
        "Session",
        ["R", "Q", "FP1", "FP2", "FP3"],
        format_func=lambda s: {
            "R": "Race", "Q": "Qualifying",
            "FP1": "Practice 1", "FP2": "Practice 2", "FP3": "Practice 3"
        }.get(s, s)
    )

# ── Load session ──────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_session_cached(year, round_number, session_type):
    try:
        session = fastf1.get_session(year, round_number, session_type)
        session.load(laps=True, telemetry=True, weather=False, messages=False)
        return session
    except Exception as e:
        return None

if st.button("Load Session", type="primary"):
    with st.spinner(f"Loading {year} R{selected_round} ({session_type})..."):
        session = load_session_cached(year, selected_round, session_type)

    if session is None:
        st.error("Could not load session. Race may not have happened yet.")
        st.stop()

    st.session_state["telemetry_session"] = session
    st.success(f"Loaded: {session.event['EventName']} {year}")

if "telemetry_session" not in st.session_state:
    st.info("Select a session above and click **Load Session** to begin.")
    st.stop()

session = st.session_state["telemetry_session"]
laps    = session.laps

# Get driver list from session
try:
    driver_list = sorted(laps["Driver"].unique().tolist())
except Exception:
    st.error("No lap data available for this session.")
    st.stop()

# Map car numbers to codes if needed
reverse_car = {str(v): k for k, v in CAR_NUMBERS_2026.items()}
driver_display = {}
for d in driver_list:
    info = DRIVERS_2026.get(d, None)
    if info:
        driver_display[d] = f"{d} — {info[0]}"
    else:
        driver_display[d] = d

st.divider()

# ── Tab layout ────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "Speed & Telemetry Traces",
    "Lap Time Comparison",
    "Tyre Strategy",
])

# ── TAB 1: Telemetry traces ───────────────────────────────────────────────
with tab1:
    st.subheader("Driver Telemetry Comparison")
    st.caption(
        "Speed, throttle, brake and gear vs lap distance. "
        "Based on each driver's fastest lap."
    )

    col_d1, col_d2 = st.columns(2)
    with col_d1:
        driver1 = st.selectbox(
            "Driver 1",
            options=driver_list,
            format_func=lambda d: driver_display.get(d, d),
            key="telem_d1",
        )
    with col_d2:
        default_d2 = driver_list[1] if len(driver_list) > 1 else driver_list[0]
        driver2 = st.selectbox(
            "Driver 2",
            options=driver_list,
            format_func=lambda d: driver_display.get(d, d),
            index=1,
            key="telem_d2",
        )

    if driver1 == driver2:
        st.warning("Select two different drivers.")
        st.stop()

    try:
        lap1 = laps.pick_drivers(driver1).pick_fastest()
        lap2 = laps.pick_drivers(driver2).pick_fastest()
        tel1 = lap1.get_telemetry().add_distance()
        tel2 = lap2.get_telemetry().add_distance()

        col1 = TEAM_COLOURS.get(
            DRIVERS_2026.get(driver1, (None, "Unknown"))[1], "#3671C6"
        )
        col2 = TEAM_COLOURS.get(
            DRIVERS_2026.get(driver2, (None, "Unknown"))[1], "#E8002D"
        )

        name1 = DRIVERS_2026.get(driver1, (driver1,))[0]
        name2 = DRIVERS_2026.get(driver2, (driver2,))[0]

        # Build 4-panel subplot
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            subplot_titles=["Speed (km/h)", "Throttle (%)",
                            "Brake", "Gear"],
            vertical_spacing=0.06,
        )

        for row, col_name, multiply in [
            (1, "Speed",    1),
            (2, "Throttle", 1),
            (3, "Brake",    1),
            (4, "nGear",    1),
        ]:
            if col_name not in tel1.columns:
                continue

            fig.add_trace(go.Scatter(
                x=tel1["Distance"], y=tel1[col_name] * multiply,
                name=name1.split()[-1], line=dict(color=col1, width=2),
                showlegend=(row == 1),
            ), row=row, col=1)

            fig.add_trace(go.Scatter(
                x=tel2["Distance"], y=tel2[col_name] * multiply,
                name=name2.split()[-1], line=dict(color=col2, width=2),
                showlegend=(row == 1),
            ), row=row, col=1)

        fig.update_layout(
            height=700,
            title_text=f"Fastest Lap Telemetry — {name1.split()[-1]} vs {name2.split()[-1]}",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", y=1.02),
        )
        fig.update_xaxes(title_text="Distance (m)", row=4, col=1)

        st.plotly_chart(fig, use_container_width=True)

        # Lap time delta
        time1 = lap1["LapTime"]
        time2 = lap2["LapTime"]
        if pd.notna(time1) and pd.notna(time2):
            delta = (time1 - time2).total_seconds()
            faster = name1.split()[-1] if delta < 0 else name2.split()[-1]
            st.metric(
                "Fastest lap delta",
                f"{abs(delta):.3f}s",
                f"{faster} was faster",
            )

    except Exception as e:
        st.error(f"Could not load telemetry: {e}")
        st.caption("Telemetry may not be available for this session/driver.")

# ── TAB 2: Lap time comparison ────────────────────────────────────────────
with tab2:
    st.subheader("Race Lap Time Comparison")
    st.caption("All lap times across the race. SC/pit laps shown as dots, racing laps as lines.")

    selected_drivers = st.multiselect(
        "Select drivers",
        options=driver_list,
        default=driver_list[:5],
        format_func=lambda d: driver_display.get(d, d),
    )

    if not selected_drivers:
        st.info("Select at least one driver.")
    else:
        fig = go.Figure()
        for drv in selected_drivers:
            drv_laps = laps.pick_drivers(drv).copy()
            if drv_laps.empty:
                continue

            drv_laps = drv_laps.dropna(subset=["LapTime"])
            drv_laps["LapTimeSec"] = drv_laps["LapTime"].dt.total_seconds()

            colour = TEAM_COLOURS.get(
                DRIVERS_2026.get(drv, (None, "Unknown"))[1], "#888888"
            )
            name = DRIVERS_2026.get(drv, (drv,))[0].split()[-1]

            # Separate clean laps from outliers
            median_time = drv_laps["LapTimeSec"].median()
            clean  = drv_laps[drv_laps["LapTimeSec"] < median_time * 1.12]
            outlier = drv_laps[drv_laps["LapTimeSec"] >= median_time * 1.12]

            fig.add_trace(go.Scatter(
                x=clean["LapNumber"], y=clean["LapTimeSec"],
                mode="lines+markers",
                name=name,
                line=dict(color=colour, width=2),
                marker=dict(size=4),
            ))

            if not outlier.empty:
                fig.add_trace(go.Scatter(
                    x=outlier["LapNumber"], y=outlier["LapTimeSec"],
                    mode="markers",
                    name=f"{name} (pit/SC)",
                    marker=dict(color=colour, size=6, symbol="x"),
                    showlegend=False,
                ))

        fig.update_layout(
            title="Lap Times by Driver",
            xaxis_title="Lap Number",
            yaxis_title="Lap Time (seconds)",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=480,
            legend=dict(orientation="h", y=-0.15),
        )
        st.plotly_chart(fig, use_container_width=True)

# ── TAB 3: Tyre strategy ──────────────────────────────────────────────────
with tab3:
    st.subheader("Tyre Strategy")
    st.caption("Which compounds were used and when each driver pitted.")

    COMPOUND_COLOURS = {
        "SOFT":   "#E8002D",
        "MEDIUM": "#FFF200",
        "HARD":   "#CCCCCC",
        "INTER":  "#39B54A",
        "WET":    "#0067FF",
        "UNKNOWN":"#888888",
    }

    if "Compound" not in laps.columns or "Stint" not in laps.columns:
        st.info("Compound/stint data not available for this session.")
    else:
        strategy = (
            laps.dropna(subset=["Compound"])
            .groupby(["Driver", "Stint", "Compound"])
            .agg(StartLap=("LapNumber", "min"), EndLap=("LapNumber", "max"))
            .reset_index()
        )

        fig = go.Figure()
        drivers_ordered = sorted(strategy["Driver"].unique())
        seen_compounds  = set()

        for _, row in strategy.iterrows():
            compound = str(row["Compound"]).upper()
            colour   = COMPOUND_COLOURS.get(compound, "#888888")
            name     = DRIVERS_2026.get(row["Driver"], (row["Driver"],))[0].split()[-1]

            show_legend = compound not in seen_compounds
            seen_compounds.add(compound)

            fig.add_trace(go.Bar(
                y=[row["Driver"]],
                x=[row["EndLap"] - row["StartLap"] + 1],
                orientation="h",
                base=row["StartLap"] - 1,
                marker_color=colour,
                marker_line=dict(width=0.5, color="black"),
                name=compound,
                showlegend=show_legend,
                width=0.6,
                hovertemplate=(
                    f"<b>{name}</b><br>"
                    f"Compound: {compound}<br>"
                    f"Laps {int(row['StartLap'])}-{int(row['EndLap'])}<br>"
                    f"Stint length: {int(row['EndLap']-row['StartLap']+1)} laps"
                    "<extra></extra>"
                ),
            ))

        fig.update_layout(
            barmode="stack",
            title="Race Tyre Strategy",
            xaxis_title="Lap Number",
            yaxis_title="Driver",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=max(400, len(drivers_ordered) * 26),
            legend=dict(orientation="h", y=-0.12),
        )
        st.plotly_chart(fig, use_container_width=True)
