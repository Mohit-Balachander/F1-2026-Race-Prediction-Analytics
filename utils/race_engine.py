"""
utils/race_engine.py -- Race telemetry engine.

Extracts frame-by-frame driver positions, track geometry,
SC simulation, and DRS zones from FastF1 sessions.
Outputs pure Python/NumPy data structures -- no Arcade, no
separate window. All rendering is done in Plotly/Streamlit.
"""

import os
import pickle
import numpy as np
import pandas as pd
import fastf1
from scipy.spatial import cKDTree

from config import CACHE_DIR, TEAM_COLOURS, DRIVERS_2026, CAR_NUMBERS_2026

fastf1.Cache.enable_cache(CACHE_DIR)

FPS = 10          # frames per second (lower = faster to compute, still smooth)
DT  = 1.0 / FPS

COMPUTED_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "computed")
os.makedirs(COMPUTED_DIR, exist_ok=True)


# ── 1. Load a session with full telemetry ────────────────────────────────
def load_race_session(year: int, round_number: int, session_type: str = "R"):
    """Load and return a FastF1 session with telemetry enabled."""
    try:
        session = fastf1.get_session(year, round_number, session_type)
        session.load(laps=True, telemetry=True, weather=True, messages=False)
        return session
    except Exception as e:
        print(f"Session load failed: {e}")
        return None


# ── 2. Build track geometry from fastest lap ─────────────────────────────
def build_track_geometry(session, track_width: int = 180) -> dict:
    """
    Build track outline (inner/outer walls) and DRS zones
    from the session's fastest lap telemetry.
    Returns dict with all track geometry arrays.
    """
    try:
        fastest = session.laps.pick_fastest()
        tel     = fastest.get_telemetry().add_distance()
    except Exception as e:
        print(f"Track geometry failed: {e}")
        return {}

    x_ref = tel["X"].to_numpy(dtype=float)
    y_ref = tel["Y"].to_numpy(dtype=float)
    dist  = tel["Distance"].to_numpy(dtype=float)
    drs   = tel["DRS"].to_numpy()

    # Compute tangent vectors
    dx   = np.gradient(x_ref)
    dy   = np.gradient(y_ref)
    norm = np.sqrt(dx**2 + dy**2)
    norm[norm == 0] = 1.0
    dx /= norm
    dy /= norm

    # Normal vectors (perpendicular to track)
    nx =  -dy
    ny =   dx

    # Inner and outer walls
    half = track_width / 2
    x_outer = x_ref + nx * half
    y_outer = y_ref + ny * half
    x_inner = x_ref - nx * half
    y_inner = y_ref - ny * half

    # DRS zones
    drs_zones = _extract_drs_zones(x_ref, y_ref, drs)

    return {
        "x_ref":   x_ref,
        "y_ref":   y_ref,
        "dist":    dist,
        "x_inner": x_inner,
        "y_inner": y_inner,
        "x_outer": x_outer,
        "y_outer": y_outer,
        "x_min":   float(min(x_ref.min(), x_inner.min(), x_outer.min())),
        "x_max":   float(max(x_ref.max(), x_inner.max(), x_outer.max())),
        "y_min":   float(min(y_ref.min(), y_inner.min(), y_outer.min())),
        "y_max":   float(max(y_ref.max(), y_inner.max(), y_outer.max())),
        "drs_zones": drs_zones,
        "total_dist": float(dist.max()),
    }


def _extract_drs_zones(x_arr, y_arr, drs_arr) -> list:
    """Identify DRS activation zones along the track."""
    zones     = []
    drs_start = None

    for i, val in enumerate(drs_arr):
        if int(val) in [10, 12, 14]:
            if drs_start is None:
                drs_start = i
        else:
            if drs_start is not None:
                zones.append({
                    "start_x": float(x_arr[drs_start]),
                    "start_y": float(y_arr[drs_start]),
                    "end_x":   float(x_arr[i - 1]),
                    "end_y":   float(y_arr[i - 1]),
                })
                drs_start = None

    if drs_start is not None:
        zones.append({
            "start_x": float(x_arr[drs_start]),
            "start_y": float(y_arr[drs_start]),
            "end_x":   float(x_arr[-1]),
            "end_y":   float(y_arr[-1]),
        })

    return zones


# ── 3. Build race frames (positions per driver per frame) ────────────────
def build_race_frames(session, track_geom: dict, max_frames: int = 3000) -> list:
    """
    Build a list of frame dicts, each containing every driver's X/Y
    position, lap number, tyre compound, speed, gear, DRS.

    Steps:
      1. Extract telemetry per driver
      2. Resample onto a common timeline at FPS rate
      3. Sort drivers by race position (laps + distance)
      4. Compute SC positions from track_status

    Returns list of frame dicts compatible with Plotly animation.
    """
    drivers     = session.drivers
    driver_data = {}
    global_tmin = None
    global_tmax = None

    for drv_no in drivers:
        try:
            code      = session.get_driver(drv_no)["Abbreviation"]
            drv_laps  = session.laps.pick_drivers(drv_no)
            if drv_laps.empty:
                continue

            t_all, x_all, y_all, d_all = [], [], [], []
            lap_all, tyre_all = [], []
            spd_all, gear_all, drs_all = [], [], []
            total_dist = 0.0

            for _, lap in drv_laps.iterlaps():
                try:
                    lt = lap.get_telemetry()
                except Exception:
                    continue
                if lt.empty:
                    continue

                if "SessionTime" not in lt.columns or "X" not in lt.columns:
                    continue

                t   = lt["SessionTime"].dt.total_seconds().to_numpy()
                x   = lt["X"].to_numpy(dtype=float)
                y   = lt["Y"].to_numpy(dtype=float)
                d   = lt["Distance"].to_numpy(dtype=float) + total_dist
                spd = lt["Speed"].to_numpy(dtype=float)   if "Speed"   in lt.columns else np.zeros(len(t))
                gr  = lt["nGear"].to_numpy(dtype=float)   if "nGear"   in lt.columns else np.zeros(len(t))
                drs = lt["DRS"].to_numpy(dtype=float)     if "DRS"     in lt.columns else np.zeros(len(t))

                tyre_int = _tyre_to_int(lap.Compound)
                ln       = int(lap.LapNumber)

                t_all.append(t);   x_all.append(x);  y_all.append(y)
                d_all.append(d);   lap_all.append(np.full(len(t), ln))
                tyre_all.append(np.full(len(t), tyre_int))
                spd_all.append(spd); gear_all.append(gr); drs_all.append(drs)

                if len(d) > 0:
                    total_dist = float(d[-1])

            if not t_all:
                continue

            t   = np.concatenate(t_all)
            x   = np.concatenate(x_all)
            y   = np.concatenate(y_all)
            d   = np.concatenate(d_all)
            lap = np.concatenate(lap_all)
            tyr = np.concatenate(tyre_all)
            spd = np.concatenate(spd_all)
            gr  = np.concatenate(gear_all)
            drs = np.concatenate(drs_all)

            order = np.argsort(t)
            t, x, y, d, lap, tyr, spd, gr, drs = (
                arr[order] for arr in [t, x, y, d, lap, tyr, spd, gr, drs]
            )

            tmin, tmax = float(t.min()), float(t.max())
            global_tmin = tmin if global_tmin is None else min(global_tmin, tmin)
            global_tmax = tmax if global_tmax is None else max(global_tmax, tmax)

            driver_data[code] = dict(t=t, x=x, y=y, d=d, lap=lap,
                                     tyre=tyr, speed=spd, gear=gr, drs=drs)
        except Exception as e:
            print(f"  Driver {drv_no} failed: {e}")

    if not driver_data:
        return []

    # Resample onto common timeline
    raw_timeline = np.arange(global_tmin, global_tmax, DT)
    # Limit frames
    if len(raw_timeline) > max_frames:
        step     = len(raw_timeline) // max_frames
        timeline = raw_timeline[::step]
    else:
        timeline = raw_timeline

    t_shifted = timeline - global_tmin
    resampled = {}
    for code, data in driver_data.items():
        t_rel  = data["t"] - global_tmin
        order  = np.argsort(t_rel)
        t_s    = t_rel[order]
        resampled[code] = {
            "x":     np.interp(t_shifted, t_s, data["x"][order]),
            "y":     np.interp(t_shifted, t_s, data["y"][order]),
            "d":     np.interp(t_shifted, t_s, data["d"][order]),
            "lap":   np.round(np.interp(t_shifted, t_s, data["lap"][order])).astype(int),
            "tyre":  np.round(np.interp(t_shifted, t_s, data["tyre"][order])).astype(int),
            "speed": np.interp(t_shifted, t_s, data["speed"][order]),
            "gear":  np.round(np.interp(t_shifted, t_s, data["gear"][order])).astype(int),
            "drs":   np.interp(t_shifted, t_s, data["drs"][order]),
        }

    # Build track_statuses for SC detection
    track_statuses = _parse_track_status(session, global_tmin)

    # Build frame list
    frames = []
    codes  = list(resampled.keys())
    n      = len(t_shifted)

    for i in range(n):
        drivers_frame = {}
        for code in codes:
            r = resampled[code]
            drivers_frame[code] = {
                "x":     float(r["x"][i]),
                "y":     float(r["y"][i]),
                "d":     float(r["d"][i]),
                "lap":   int(r["lap"][i]),
                "tyre":  int(r["tyre"][i]),
                "speed": float(r["speed"][i]),
                "gear":  int(r["gear"][i]),
                "drs":   int(r["drs"][i]),
            }

        # Sort by race position
        sorted_drivers = sorted(
            drivers_frame.items(),
            key=lambda kv: (kv[1]["lap"], kv[1]["d"]),
            reverse=True
        )
        for pos, (code, _) in enumerate(sorted_drivers, 1):
            drivers_frame[code]["position"] = pos

        frames.append({
            "t":       round(float(t_shifted[i]), 2),
            "drivers": drivers_frame,
        })

    # Compute SC positions
    _compute_safety_car_positions(frames, track_statuses, track_geom)

    return frames


# ── 4. Safety Car position simulation ───────────────────────────────────
def _compute_safety_car_positions(frames: list, track_statuses: list, track_geom: dict):
    """
    Simulate SC positions per frame.
    SC has no GPS in F1 API -- we simulate:
      - Deploying: SC animates from pit exit onto track
      - On track: SC leads at 150m ahead of race leader
      - Returning: SC accelerates then enters pit lane

    Adapted from IAmTomShaw/f1-race-replay SC simulation logic.
    """
    if not frames or not track_geom:
        return

    x_ref  = track_geom.get("x_ref")
    y_ref  = track_geom.get("y_ref")
    if x_ref is None or len(x_ref) < 10:
        return

    # Dense reference polyline
    t_old = np.linspace(0, 1, len(x_ref))
    t_new = np.linspace(0, 1, 4000)
    xd    = np.interp(t_new, t_old, x_ref)
    yd    = np.interp(t_new, t_old, y_ref)

    diffs     = np.sqrt(np.diff(xd)**2 + np.diff(yd)**2)
    cumdist   = np.concatenate(([0.0], np.cumsum(diffs)))
    total_len = float(cumdist[-1])
    tree      = cKDTree(np.column_stack([xd, yd]))

    def pos_at(d):
        d  = d % total_len
        i  = int(np.searchsorted(cumdist, d))
        i  = min(i, len(xd) - 1)
        return float(xd[i]), float(yd[i])

    def dist_of(x, y):
        _, i = tree.query([x, y])
        return float(cumdist[i])

    # Normals for pit lane offset
    dx    = np.gradient(xd)
    dy    = np.gradient(yd)
    norm  = np.sqrt(dx**2 + dy**2)
    norm[norm == 0] = 1.0
    nx, ny = -dy / norm, dx / norm

    PIT_OFFSET = 400
    pit_exit_d = total_len * 0.05
    pit_exit_idx = int(np.searchsorted(cumdist, pit_exit_d))
    pit_exit_tx, pit_exit_ty = float(xd[pit_exit_idx]), float(yd[pit_exit_idx])
    pit_exit_px  = pit_exit_tx + nx[pit_exit_idx] * PIT_OFFSET
    pit_exit_py  = pit_exit_ty + ny[pit_exit_idx] * PIT_OFFSET

    pit_entry_d  = total_len * 0.95
    pit_entry_idx = int(np.searchsorted(cumdist, pit_entry_d))
    pit_entry_tx, pit_entry_ty = float(xd[pit_entry_idx]), float(yd[pit_entry_idx])
    pit_entry_px  = pit_entry_tx + nx[pit_entry_idx] * PIT_OFFSET
    pit_entry_py  = pit_entry_ty + ny[pit_entry_idx] * PIT_OFFSET

    # Identify SC periods
    sc_periods = [s for s in track_statuses if str(s.get("status")) == "4"]
    if not sc_periods:
        return

    SC_OFFSET_M      = 150.0
    DEPLOY_DURATION  = 4.0
    RETURN_ACCEL     = 5.0
    RETURN_PIT       = 3.0
    RETURN_TOTAL     = RETURN_ACCEL + RETURN_PIT

    sc_state = {}

    for fi, frame in enumerate(frames):
        t = frame["t"]
        active_sc = active_idx = None
        for sci, sc in enumerate(sc_periods):
            s, e = sc["start_time"], sc.get("end_time")
            eff_end = (e + RETURN_TOTAL) if e else None
            if t >= s and (eff_end is None or t < eff_end):
                active_sc, active_idx = sc, sci
                break

        if active_sc is None:
            frame["safety_car"] = None
            continue

        sc_start = active_sc["start_time"]
        sc_end   = active_sc.get("end_time")
        elapsed  = t - sc_start

        if active_idx not in sc_state:
            sc_state[active_idx] = {
                "track_dist": pit_exit_d,
                "last_t": t,
            }

        state  = sc_state[active_idx]
        dt_frm = max(0.0, t - state["last_t"])
        state["last_t"] = t

        if elapsed < DEPLOY_DURATION:
            # Deploying: interpolate from pit to track
            prog   = elapsed / DEPLOY_DURATION
            smooth = 0.5 - 0.5 * np.cos(prog * np.pi)
            sc_x   = pit_exit_px + smooth * (pit_exit_tx - pit_exit_px)
            sc_y   = pit_exit_py + smooth * (pit_exit_ty - pit_exit_py)
            phase, alpha = "deploying", prog

        elif sc_end is not None and t >= sc_end:
            # Returning
            ret_elapsed = t - sc_end
            if ret_elapsed < RETURN_ACCEL:
                state["track_dist"] = (state["track_dist"] + 400.0 * dt_frm) % total_len
                sc_x, sc_y = pos_at(state["track_dist"])
                phase, alpha = "returning", 1.0
            else:
                pit_prog = min(1.0, (ret_elapsed - RETURN_ACCEL) / RETURN_PIT)
                smooth   = 0.5 - 0.5 * np.cos(pit_prog * np.pi)
                tx, ty   = pos_at(state["track_dist"])
                sc_x = tx + smooth * (pit_entry_px - tx)
                sc_y = ty + smooth * (pit_entry_py - ty)
                phase, alpha = "returning", max(0.0, 1.0 - pit_prog)
        else:
            # On track: lead the race leader
            leader_x = leader_y = None
            best_prog = -1
            for code, pos in frame["drivers"].items():
                prog = (pos["lap"] - 1) * total_len + pos["d"]
                if prog > best_prog:
                    best_prog = prog
                    leader_x, leader_y = pos["x"], pos["y"]

            if leader_x is not None:
                leader_d = dist_of(leader_x, leader_y)
                target_d = (leader_d + SC_OFFSET_M) % total_len
                state["track_dist"] = target_d
            else:
                state["track_dist"] = (state["track_dist"] + 50.0 * dt_frm) % total_len

            sc_x, sc_y = pos_at(state["track_dist"])
            phase, alpha = "on_track", 1.0

        frame["safety_car"] = {
            "x": round(sc_x, 1), "y": round(sc_y, 1),
            "phase": phase, "alpha": round(alpha, 3),
        }


# ── 5. Track status parsing ──────────────────────────────────────────────
def _parse_track_status(session, global_tmin: float) -> list:
    """Parse session.track_status into a list of status dicts."""
    try:
        ts = session.track_status
    except Exception:
        return []

    result = []
    for row in ts.to_dict("records"):
        start = row["Time"].total_seconds() - global_tmin
        if result:
            result[-1]["end_time"] = start
        result.append({"status": str(row["Status"]), "start_time": start, "end_time": None})
    return result


# ── 6. Extract race events for timeline display ──────────────────────────
def extract_race_events(frames: list, track_statuses: list) -> list:
    """
    Build a list of notable race events (SC, VSC, yellow/red flags, DNFs)
    for display on a race progress timeline.
    """
    events = []
    n      = len(frames)
    fps    = FPS

    # Flag events from track_status
    STATUS_MAP = {
        "2": "yellow_flag",
        "4": "safety_car",
        "5": "red_flag",
        "6": "vsc",
        "7": "vsc",
    }
    for s in track_statuses:
        etype = STATUS_MAP.get(str(s.get("status", "")))
        if not etype:
            continue
        start_frame = int(s["start_time"] * fps)
        end_frame   = int(s["end_time"]   * fps) if s.get("end_time") else start_frame + fps * 10
        events.append({
            "type":       etype,
            "frame":      max(0, start_frame),
            "end_frame":  min(n, end_frame),
            "label":      "",
        })

    # DNF detection (drivers who disappear from frames)
    prev_drivers = set()
    for i in range(0, n, fps):
        curr = set(frames[i]["drivers"].keys())
        for drv in prev_drivers - curr:
            prev_f  = frames[max(0, i - fps)]
            drv_inf = prev_f["drivers"].get(drv, {})
            events.append({
                "type":  "dnf",
                "frame": i,
                "label": drv,
                "lap":   drv_inf.get("lap", "?"),
            })
        prev_drivers = curr

    return events


# ── 7. Cache helpers ─────────────────────────────────────────────────────
def _cache_path(year: int, round_number: int, session_type: str) -> str:
    return os.path.join(COMPUTED_DIR, f"{year}_R{round_number}_{session_type}_replay.pkl")


def load_cached(year, round_number, session_type="R") -> dict:
    path = _cache_path(year, round_number, session_type)
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception:
            pass
    return {}


def save_cache(data: dict, year, round_number, session_type="R"):
    path = _cache_path(year, round_number, session_type)
    try:
        with open(path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        print(f"Cache save failed: {e}")


# ── 8. Top-level: get all replay data for a race ─────────────────────────
def get_replay_data(year: int, round_number: int,
                    session_type: str = "R",
                    force_refresh: bool = False) -> dict:
    """
    Main entry point. Returns all data needed for the Plotly replay:
      - track_geom: geometry arrays
      - frames:     list of frame dicts
      - events:     race events for progress bar
      - driver_colours: {code: hex}
      - total_laps: int

    Caches to disk after first computation.
    """
    if not force_refresh:
        cached = load_cached(year, round_number, session_type)
        if cached:
            print(f"Loaded cached replay data for {year} R{round_number}")
            return cached

    session = load_race_session(year, round_number, session_type)
    if session is None:
        return {}

    print(f"Computing replay data for {year} R{round_number}...")

    track_geom = build_track_geometry(session)
    if not track_geom:
        return {}

    frames = build_race_frames(session, track_geom, max_frames=2000)
    if not frames:
        return {}

    track_statuses = _parse_track_status(
        session,
        session.laps["SessionTime"].dt.total_seconds().min()
        if (not session.laps.empty and "SessionTime" in session.laps.columns)
        else 0.0
    )
    events = extract_race_events(frames, track_statuses)

    # Driver colours from FastF1
    try:
        colour_map = fastf1.plotting.get_driver_color_mapping(session)
    except Exception:
        colour_map = {}

    total_laps = int(session.laps["LapNumber"].max()) if not session.laps.empty else 0

    data = {
        "track_geom":      track_geom,
        "frames":          frames,
        "events":          events,
        "driver_colours":  colour_map,
        "total_laps":      total_laps,
        "event_name":      session.event["EventName"],
        "year":            year,
        "round":           round_number,
    }

    save_cache(data, year, round_number, session_type)
    print(f"Replay data ready: {len(frames)} frames, {len(frames[0]['drivers'])} drivers")
    return data


# ── 9. Tyre integer mapping ──────────────────────────────────────────────
TYRE_MAP = {"SOFT": 1, "MEDIUM": 2, "HARD": 3, "INTER": 4, "WET": 5}
TYRE_COLOURS = {
    1: "#E8002D",   # Soft  - red
    2: "#FFF200",   # Medium - yellow
    3: "#CCCCCC",   # Hard  - white/gray
    4: "#39B54A",   # Inter - green
    5: "#0067FF",   # Wet   - blue
    0: "#888888",   # Unknown
}

def _tyre_to_int(compound) -> int:
    if pd.isna(compound):
        return 0
    return TYRE_MAP.get(str(compound).upper(), 0)


def tyre_colour(tyre_int: int) -> str:
    return TYRE_COLOURS.get(int(tyre_int), "#888888")


# ── 10. Driver colour helper ─────────────────────────────────────────────
def get_driver_colour(code: str, driver_colours: dict) -> str:
    """Get hex colour for a driver, falling back to team colour."""
    raw = driver_colours.get(code)
    if raw:
        if isinstance(raw, str) and raw.startswith("#"):
            return raw
        if isinstance(raw, (tuple, list)) and len(raw) >= 3:
            return "#{:02X}{:02X}{:02X}".format(int(raw[0]), int(raw[1]), int(raw[2]))
    team = DRIVERS_2026.get(code, (None, "Unknown"))[1]
    return TEAM_COLOURS.get(team, "#888888")