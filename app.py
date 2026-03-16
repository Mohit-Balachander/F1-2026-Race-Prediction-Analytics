"""
app.py -- 2026 F1 Season Prediction Dashboard

Run with:  streamlit run app.py

Pages:
  1. Current Standings  -- live 2026 results + championship table
  2. Next Race Predictor -- predict the upcoming GP (Japan R3 next)
  3. Championship Forecast -- project final standings for all 22 races
  4. Model Insights      -- feature importance, accuracy, era weighting
  5. Add Race Results    -- manually enter new results as season progresses
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="F1 2026 Predictor",
    page_icon="F1",
    layout="wide",
    initial_sidebar_state="expanded",
)

from config import (
    DRIVERS_2026, CALENDAR_2026, TEAM_COLOURS, DRIVER_COLOURS,
    RESULTS_2026, ERA_WEIGHTS,
)
from utils.data_loader import (
    build_full_dataset, compute_rolling_stats,
    compute_standings, load_2026_real_data,
)
from models.predictor import train_model, predict_race, forecast_championship

# ── Sidebar ────────────────────────────────────────────────────────────────
st.sidebar.title("F1 2026 PREDICTOR")
st.sidebar.caption("Train on 2023/2024 history. Predict 2026.")
st.sidebar.divider()

page = st.sidebar.radio("Navigate", [
    "Current Standings",
    "Next Race Predictor",
    "Championship Forecast",
    "Race Replay + Prediction",
    "Model Insights",
    "Add Race Result",
])

st.sidebar.divider()
st.sidebar.caption(
    "**Data:** FastF1 (2023-2024) + Jolpica API (2026)  \n"
    "**Model:** GradientBoosting + RandomForest ensemble  \n"
    "**Era weight:** 2026 x2.5 | 2024 x0.6 | 2023 x0.4"
)

# ── Session state: train model once ───────────────────────────────────────
if "model_ready" not in st.session_state:
    st.session_state["model_ready"] = False
if "model_obj" not in st.session_state:
    st.session_state["model_obj"] = None
if "rolling_df" not in st.session_state:
    st.session_state["rolling_df"] = None
if "extra_results" not in st.session_state:
    st.session_state["extra_results"] = {}
if "real_2026_df" not in st.session_state:
    st.session_state["real_2026_df"] = pd.DataFrame()


def get_combined_results():
    """Use real 2026 FastF1 data if loaded, else hardcoded config."""
    if "real_2026_df" in st.session_state and not st.session_state["real_2026_df"].empty:
        # Build dict from real FastF1 data for compatibility
        real = st.session_state["real_2026_df"][st.session_state["real_2026_df"]["Year"] == 2026]
        combined = {}
        for rnd in real["Round"].unique():
            rnd_df = real[real["Round"] == rnd]
            combined[int(rnd)] = {
                row["Driver"]: int(row["FinishPos"])
                for _, row in rnd_df.iterrows()
            }
        combined.update(st.session_state.get("extra_results", {}))
        return combined
    # Fallback to hardcoded
    combined = dict(RESULTS_2026)
    combined.update(st.session_state.get("extra_results", {}))
    return combined


def get_remaining_rounds():
    combined = get_combined_results()
    completed = set(combined.keys())
    return [r for r in CALENDAR_2026 if r not in completed]


# ===========================================================================
# PAGE 1: CURRENT STANDINGS
# ===========================================================================
if page == "Current Standings":
    st.title("2026 F1 Championship — Current Standings")

    results = get_combined_results()
    completed_rounds = sorted(results.keys())
    remaining_rounds = get_remaining_rounds()

    # ── Season progress bar ────────────────────────────────────────────────
    total = len(CALENDAR_2026)
    done  = len(completed_rounds)
    st.progress(done / total, text=f"Season progress: {done}/{total} races completed")

    # Next race info
    if remaining_rounds:
        next_rnd  = remaining_rounds[0]
        next_info = CALENDAR_2026[next_rnd]
        st.info(f"**Next:** R{next_rnd} — {next_info[0]} GP | {next_info[1]} | {next_info[2]}")

    st.divider()

    # ── Championship standings ──────────────────────────────────────────────
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Driver Championship")
        standings = compute_standings(results)

        # Trophy emoji for top 3
        standings["Pos"] = standings.index
        standings["Pos"] = standings["Pos"].apply(
            lambda p: {1: "1st", 2: "2nd", 3: "3rd"}.get(p, f"{p}th")
        )

        st.dataframe(
            standings[["Pos", "Name", "Team", "Points"]],
            use_container_width=True,
            hide_index=True,
            height=580,
        )

    with col2:
        st.subheader("Constructor Championship")
        con_pts = standings.groupby("Team")["Points"].sum()\
                           .reset_index().sort_values("Points", ascending=False)
        con_pts.index = range(1, len(con_pts) + 1)
        con_pts.index.name = "Pos"
        st.dataframe(con_pts, use_container_width=True, height=380)

        # Pie chart
        fig_pie = px.pie(
            con_pts,
            names="Team",
            values="Points",
            color="Team",
            color_discrete_map=TEAM_COLOURS,
            title="Points share by constructor",
            hole=0.4,
        )
        fig_pie.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            height=250,
            showlegend=False,
            margin=dict(t=40, b=0, l=0, r=0),
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    st.divider()

    # ── Race-by-race results grid ──────────────────────────────────────────
    st.subheader("Race Results Grid")
    if completed_rounds:
        grid_rows = []
        for drv_code, info in DRIVERS_2026.items():
            row = {"Driver": drv_code, "Name": info[0], "Team": info[1]}
            pts_total = 0
            for rnd in completed_rounds:
                pos = results[rnd].get(drv_code, "-")
                if isinstance(pos, int) and pos <= 20:
                    pts = {1:25, 2:18, 3:15, 4:12, 5:10, 6:8, 7:6, 8:4, 9:2, 10:1}.get(pos, 0)
                    pts_total += pts
                    row[f"R{rnd}"] = f"P{pos}"
                elif isinstance(pos, int) and pos == 21:
                    row[f"R{rnd}"] = "DNF"
                elif isinstance(pos, int) and pos >= 22:
                    row[f"R{rnd}"] = "DNS"
                else:
                    row[f"R{rnd}"] = "-"
            row["Total"] = pts_total
            grid_rows.append(row)

        grid_df = pd.DataFrame(grid_rows).sort_values("Total", ascending=False)
        st.dataframe(grid_df, use_container_width=True, hide_index=True)


# ===========================================================================
# PAGE 2: NEXT RACE PREDICTOR
# ===========================================================================
elif page == "Next Race Predictor":
    st.title("Next Race Prediction")

    remaining = get_remaining_rounds()
    if not remaining:
        st.success("All 22 races completed!")
        st.stop()

    # ── Race selector ──────────────────────────────────────────────────────
    next_rnd = remaining[0]
    options  = {rnd: f"R{rnd} — {CALENDAR_2026[rnd][0]} GP" for rnd in remaining}

    selected_rnd = st.selectbox(
        "Select race to predict",
        options=list(options.keys()),
        format_func=lambda r: options[r],
    )
    race_info = CALENDAR_2026[selected_rnd]

    st.markdown(f"### {race_info[0]} Grand Prix | {race_info[1]} | {race_info[2]}")
    st.divider()

    # ── Optional qualifying positions ─────────────────────────────────────
    with st.expander("Optional: Enter qualifying positions (improves accuracy)"):
        st.caption("Enter grid positions if qualifying has happened.")
        qual_positions = {}
        cols = st.columns(4)
        for i, (drv_code, info) in enumerate(DRIVERS_2026.items()):
            with cols[i % 4]:
                pos = st.number_input(
                    f"{drv_code} — {info[0].split()[1]}",
                    min_value=1, max_value=22, value=i+1,
                    key=f"qual_{drv_code}",
                )
                qual_positions[drv_code] = pos

    # ── Train / load model ─────────────────────────────────────────────────
    col_btn, col_status = st.columns([1, 3])
    with col_btn:
        train_btn = st.button("Train Model & Predict", type="primary", use_container_width=True)

    if train_btn or st.session_state["model_ready"]:
        if not st.session_state["model_ready"]:
            with st.spinner("Loading data via FastF1 v3.8.1... (first run: 3-5 min, cached after)"):
                progress_placeholder = st.empty()

                def update_progress(msg):
                    progress_placeholder.caption(msg)

                # Single call loads everything: 2023+2024 history + real 2026 data
                combined = build_full_dataset(
                    max_historical_rounds=10,
                    progress_callback=update_progress,
                )

                # Store real 2026 slice separately for standings
                if not combined.empty:
                    st.session_state["real_2026_df"] = combined[combined["Year"] == 2026]

                st.write(f"Total rows loaded: {len(combined)} "
                         f"({len(combined[combined['Year']==2026])} from 2026)")

                rolling = compute_rolling_stats(combined)
                model   = train_model(rolling)

                st.session_state["model_obj"]   = model
                st.session_state["rolling_df"]  = rolling
                st.session_state["model_ready"] = True
                progress_placeholder.empty()

        model  = st.session_state["model_obj"]
        rolling = st.session_state["rolling_df"]

        # Build driver feature rows for prediction
        latest_features = (
            rolling.sort_values(["Driver", "Year", "Round"])
            .groupby("Driver").last()
            .reset_index()
        )

        # Add any drivers missing from rolling (new drivers / data gaps)
        known_drivers = set(latest_features["Driver"].unique())
        missing = [d for d in DRIVERS_2026 if d not in known_drivers]
        if missing:
            filler_rows = []
            for drv in missing:
                info = DRIVERS_2026[drv]
                filler_rows.append({
                    "Driver": drv, "Team": info[1],
                    "GridPosition": 15.0, "FinishPos": 15.0,
                    "AvgFinish": 15.0, "AvgGrid": 15.0,
                    "RecentForm": 2.0, "DNFRate": 0.1,
                    "MedianPaceDelta": 0.0, "PaceConsistency": 1.5,
                    "EraWeight": 2.5,
                })
            latest_features = pd.concat(
                [latest_features, pd.DataFrame(filler_rows)], ignore_index=True
            )

        # Predict
        prediction = predict_race(model, latest_features, qual_positions)

        st.divider()
        st.subheader(f"Predicted Finishing Order — {race_info[0]} GP")

        # ── Podium highlight ────────────────────────────────────────────────
        p1, p2, p3 = st.columns(3)
        top3 = prediction.head(3)

        podium_order = [
            (p2, top3.iloc[1] if len(top3) > 1 else None, "P2"),
            (p1, top3.iloc[0] if len(top3) > 0 else None, "P1 - Winner"),
            (p3, top3.iloc[2] if len(top3) > 2 else None, "P3"),
        ]

        for col, row, pos_label in podium_order:
            if row is None:
                continue
            team_col = TEAM_COLOURS.get(row["Team"], "#888")
            col.markdown(
                f"""<div style="background:{team_col}22; border:2px solid {team_col};
                border-radius:12px; padding:16px; text-align:center;">
                <div style="font-size:22px; font-weight:700;">{row['Name']}</div>
                <div style="font-size:14px; opacity:0.8;">{row['Team']}</div>
                <div style="font-size:18px; font-weight:600; margin-top:8px;">
                {pos_label}</div>
                <div style="font-size:12px; opacity:0.7;">Confidence: {row['Confidence']}</div>
                </div>""",
                unsafe_allow_html=True,
            )

        st.divider()

        # ── Full predicted grid ─────────────────────────────────────────────
        col1, col2 = st.columns([2, 1])
        with col1:
            display = prediction[["PredictedPos", "Driver", "Name", "Team",
                                   "PredictedPoints", "Confidence"]].copy()
            display.columns = ["Pos", "Code", "Driver", "Team", "Pts", "Confidence"]
            st.dataframe(display, use_container_width=True, hide_index=True, height=580)

        with col2:
            # Bar chart of predicted points
            fig = px.bar(
                prediction.head(10),
                x="PredictedPoints",
                y="Name",
                orientation="h",
                color="Team",
                color_discrete_map=TEAM_COLOURS,
                title="Predicted points (top 10)",
            )
            fig.update_layout(
                yaxis=dict(autorange="reversed"),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=400,
                showlegend=False,
                margin=dict(t=30, b=0),
            )
            st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("Click **Train Model & Predict** to generate predictions for this race.")
        st.caption(
            "First run downloads 10 races each from 2023 and 2024 via FastF1 (~2-4 minutes). "
            "Subsequent runs are instant (cached locally)."
        )


# ===========================================================================
# PAGE 3: CHAMPIONSHIP FORECAST
# ===========================================================================
elif page == "Championship Forecast":
    st.title("2026 Championship Forecast")
    st.caption(
        "Predicts all remaining 2026 races and projects final standings. "
        "Based on 2023-2024 historical training + 2026 R1-R2 results."
    )

    if not st.session_state["model_ready"]:
        st.warning("Train the model first: go to **Next Race Predictor** and click Train.")
        st.stop()

    model   = st.session_state["model_obj"]
    rolling = st.session_state["rolling_df"]
    results = get_combined_results()
    remaining = get_remaining_rounds()

    st.info(f"Simulating **{len(remaining)} remaining races** (R{remaining[0] if remaining else 'N/A'} to R22)")

    if st.button("Run Championship Forecast", type="primary"):
        with st.spinner("Forecasting all remaining races..."):
            latest_features = (
                rolling.sort_values(["Driver", "Year", "Round"])
                .groupby("Driver").last().reset_index()
            )
            # Fill missing drivers
            known = set(latest_features["Driver"])
            for drv in DRIVERS_2026:
                if drv not in known:
                    info = DRIVERS_2026[drv]
                    latest_features = pd.concat([latest_features, pd.DataFrame([{
                        "Driver": drv, "Team": info[1],
                        "GridPosition": 15.0, "FinishPos": 15.0,
                        "AvgFinish": 15.0, "AvgGrid": 15.0,
                        "RecentForm": 2.0, "DNFRate": 0.1,
                        "MedianPaceDelta": 0.0, "PaceConsistency": 1.5,
                        "EraWeight": 2.5,
                    }])], ignore_index=True)

            forecast = forecast_championship(model, latest_features, results, remaining)

        st.divider()
        st.subheader("Projected Final Championship Standings")

        col1, col2 = st.columns([1, 1])
        with col1:
            st.dataframe(forecast, use_container_width=True, height=600)

        with col2:
            fig = px.bar(
                forecast.reset_index(),
                x="ProjectedPoints",
                y="Name",
                orientation="h",
                color="Team",
                color_discrete_map=TEAM_COLOURS,
                title="Projected final points",
            )
            fig.update_layout(
                yaxis=dict(autorange="reversed"),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                showlegend=False,
                height=580,
            )
            st.plotly_chart(fig, use_container_width=True)

        st.divider()
        # Constructor forecast
        st.subheader("Projected Constructor Standings")
        con_forecast = forecast.groupby("Team")["ProjectedPoints"]\
                               .sum().reset_index()\
                               .sort_values("ProjectedPoints", ascending=False)
        con_forecast.index = range(1, len(con_forecast) + 1)

        fig2 = px.bar(
            con_forecast,
            x="Team",
            y="ProjectedPoints",
            color="Team",
            color_discrete_map=TEAM_COLOURS,
            title="Projected Constructor Championship",
        )
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
        )
        st.plotly_chart(fig2, use_container_width=True)


# ===========================================================================
# PAGE 4: MODEL INSIGHTS
# ===========================================================================
elif page == "Race Replay + Prediction":
    exec(open("pages/page_replay.py", encoding="utf-8").read())

elif page == "Model Insights":
    st.title("Model Insights")

    if not st.session_state["model_ready"]:
        st.warning("Train the model first via **Next Race Predictor**.")
        st.stop()

    model = st.session_state["model_obj"]

    col1, col2, col3 = st.columns(3)
    col1.metric("Cross-val MAE", f"{model['mae']:.3f} positions")
    col2.metric("Training samples", f"{model['n_samples']:,}")
    col3.metric("Era weight (2026)", f"{ERA_WEIGHTS.get(2026, 2.5)}x")

    st.divider()

    st.subheader("Feature Importance")
    st.caption(
        "Which features drive predictions most. "
        "GridPosition (qualifying) is typically the strongest single predictor."
    )
    fig = px.bar(
        model["importance"],
        x="Importance",
        y="Feature",
        orientation="h",
        color="Importance",
        color_continuous_scale="Blues",
        title="GradientBoosting Feature Importance",
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(autorange="reversed"),
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Era Weighting Explained")
    era_df = pd.DataFrame([
        {"Season": "2023", "Weight": ERA_WEIGHTS[2023], "Reason": "Oldest -- least relevant to 2026 regs"},
        {"Season": "2024", "Weight": ERA_WEIGHTS[2024], "Reason": "Recent but pre-reg-change"},
        {"Season": "2026", "Weight": ERA_WEIGHTS[2026], "Reason": "Current era -- highest weight"},
    ])
    st.dataframe(era_df, use_container_width=True, hide_index=True)
    st.caption(
        "Sample weights passed directly into XGBoost fit(). "
        "As more 2026 races complete, the 2026 data dominates training more strongly. "
        "This is the key mechanism for handling the regulation reset -- "
        "we down-weight historical data from a different car generation."
    )


# ===========================================================================
# PAGE 5: ADD RACE RESULT
# ===========================================================================
elif page == "Add Race Result":
    st.title("Add New Race Result")
    st.caption(
        "After each race, enter the finishing positions here. "
        "This updates the standings and re-seeds the model for future predictions."
    )

    results = get_combined_results()
    completed = set(results.keys())
    available_rounds = [r for r in CALENDAR_2026 if r not in completed]

    if not available_rounds:
        st.success("All 22 races have results entered.")
        st.stop()

    selected_rnd = st.selectbox(
        "Which race are you entering results for?",
        options=available_rounds,
        format_func=lambda r: f"R{r} — {CALENDAR_2026[r][0]} GP ({CALENDAR_2026[r][2]})",
    )

    st.subheader(f"Enter finishing positions — {CALENDAR_2026[selected_rnd][0]} GP")
    st.caption("Enter 1-20 for classified finishers. 21 = DNF. 22 = DNS.")

    new_positions = {}
    cols = st.columns(4)
    for i, (drv_code, info) in enumerate(DRIVERS_2026.items()):
        with cols[i % 4]:
            pos = st.number_input(
                f"{drv_code} — {info[0].split()[0]} {info[0].split()[-1]}",
                min_value=1, max_value=22,
                value=i + 1,
                key=f"result_{drv_code}",
            )
            new_positions[drv_code] = pos

    if st.button("Save Race Result", type="primary"):
        st.session_state["extra_results"][selected_rnd] = new_positions
        # Invalidate model so it retrains with new data
        st.session_state["model_ready"] = False
        st.success(
            f"R{selected_rnd} {CALENDAR_2026[selected_rnd][0]} GP results saved. "
            "Model will retrain automatically on next prediction."
        )
        st.balloons()