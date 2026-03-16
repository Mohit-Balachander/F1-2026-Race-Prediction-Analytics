"""
replay_window.py -- F1 Race Replay Desktop Window

Standalone PySide6 application that animates F1 car positions
on a real circuit map pulled from FastF1 telemetry.

Launch standalone:
    python replay_window.py --year 2026 --round 1
    python replay_window.py --year 2026 --round 2 --session Q

Launch from Streamlit:
    import subprocess
    subprocess.Popen(["python", "replay_window.py", "--year", "2026", "--round", "1"])

Features:
    - Animated car dots on real circuit layout
    - Track turns YELLOW during SC, RED during red flag
    - Safety Car shown as orange diamond
    - DRS zones highlighted in green
    - Live leaderboard panel on right
    - Play / Pause / Speed controls
    - Scrubber bar to seek through race
    - Click any car dot to see speed, gear, DRS, tyre
    - Works for 2023, 2024, 2026 seasons
"""

import sys
import os
import argparse
import math

# Add project root to path so we can import config, utils etc
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QPushButton, QLabel, QSlider, QTableWidget, QTableWidgetItem,
    QSizePolicy, QFrame, QComboBox, QHeaderView, QProgressDialog,
)
from PySide6.QtCore import (
    Qt, QTimer, QThread, Signal, QPointF, QRectF,
)
from PySide6.QtGui import (
    QPainter, QColor, QPen, QBrush, QFont, QPolygonF,
    QPainterPath, QLinearGradient, QFontMetrics,
)

import fastf1
import numpy as np

from config import (
    CACHE_DIR, CALENDAR_2026, DRIVERS_2026,
    TEAM_COLOURS, RESULTS_2026,
)
from utils.race_engine import (
    get_replay_data, get_driver_colour, tyre_colour, TYRE_MAP,
)

fastf1.Cache.enable_cache(CACHE_DIR)

# ── Colour constants ──────────────────────────────────────────────────────
BG_DARK      = QColor("#0d0d0d")
TRACK_NORMAL = QColor("#2a2a2a")
TRACK_SC     = QColor("#7a4000")
TRACK_VSC    = QColor("#5a4000")
TRACK_RED    = QColor("#6a0000")
TRACK_YELLOW = QColor("#5a5000")
WALL_COLOUR  = QColor("#555555")
DRS_COLOUR   = QColor("#00cc44")
SF_COLOUR    = QColor("#ffffff")
TEXT_PRIMARY = QColor("#f0f0f0")
TEXT_MUTED   = QColor("#888888")
PANEL_BG     = QColor("#111111")
SC_COLOUR    = QColor("#ff8c00")

TYRE_COLOURS_QT = {
    1: QColor("#e8002d"),   # Soft
    2: QColor("#fff200"),   # Medium
    3: QColor("#cccccc"),   # Hard
    4: QColor("#39b54a"),   # Inter
    5: QColor("#0067ff"),   # Wet
    0: QColor("#888888"),   # Unknown
}

TYRE_NAMES = {1:"SOFT", 2:"MED", 3:"HARD", 4:"INTER", 5:"WET", 0:"?"}


# ── Data loader thread ─────────────────────────────────────────────────────
class DataLoaderThread(QThread):
    finished = Signal(dict)
    progress = Signal(str)
    error    = Signal(str)

    def __init__(self, year, round_number, session_type):
        super().__init__()
        self.year         = year
        self.round_number = round_number
        self.session_type = session_type

    def run(self):
        try:
            self.progress.emit(f"Loading F1 {self.year} Round {self.round_number} "
                               f"Session '{self.session_type}'...")
            data = get_replay_data(
                self.year, self.round_number,
                self.session_type, force_refresh=False
            )
            if data:
                self.finished.emit(data)
            else:
                self.error.emit("No replay data available for this session.")
        except Exception as e:
            self.error.emit(str(e))


# ── Track canvas widget ────────────────────────────────────────────────────
class TrackCanvas(QWidget):
    """
    Renders the circuit map and animated car positions.
    All drawing is done in Qt's QPainter -- smooth and fast.
    """

    car_clicked = Signal(str)   # emits driver code on click

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(600, 500)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("background-color: #0d0d0d;")

        self.data        = None
        self.frame_idx   = 0
        self.track       = None
        self.frames      = []
        self.events      = []

        # Cached track path
        self._outer_path  = None
        self._inner_path  = None
        self._track_fill  = None
        self._transform   = None   # (scale, offset_x, offset_y)

    def set_data(self, data):
        self.data   = data
        self.track  = data["track_geom"]
        self.frames = data["frames"]
        self.events = data.get("events", [])
        self._build_track_paths()
        self.update()

    def set_frame(self, idx):
        self.frame_idx = max(0, min(idx, len(self.frames) - 1))
        self.update()

    def _build_track_paths(self):
        """Pre-build QPainterPath objects for outer/inner walls."""
        if self.track is None:
            return
        self._outer_path = self._points_to_path(
            self.track["x_outer"], self.track["y_outer"])
        self._inner_path = self._points_to_path(
            self.track["x_inner"], self.track["y_inner"])

    def _points_to_path(self, xs, ys):
        path = QPainterPath()
        if len(xs) == 0:
            return path
        path.moveTo(float(xs[0]), float(ys[0]))
        for x, y in zip(xs[1:], ys[1:]):
            path.lineTo(float(x), float(y))
        path.closeSubpath()
        return path

    def _get_transform(self):
        """Compute scale + offset to fit track into widget."""
        if self.track is None:
            return 1.0, 0.0, 0.0

        pad   = 40
        tw    = self.width()  - 2 * pad
        th    = self.height() - 2 * pad
        xspan = self.track["x_max"] - self.track["x_min"]
        yspan = self.track["y_max"] - self.track["y_min"]

        if xspan == 0 or yspan == 0:
            return 1.0, pad, pad

        scale  = min(tw / xspan, th / yspan)
        ox     = pad + (tw - xspan * scale) / 2
        oy     = pad + (th - yspan * scale) / 2
        return scale, ox, oy

    def _to_screen(self, world_x, world_y, scale, ox, oy):
        """Convert world coordinates to screen pixels."""
        sx = (world_x - self.track["x_min"]) * scale + ox
        sy = self.height() - ((world_y - self.track["y_min"]) * scale + oy)
        return sx, sy

    def _get_track_colour(self):
        """Return track surface colour based on current race status."""
        if not self.events:
            return TRACK_NORMAL
        for ev in self.events:
            sf = ev.get("frame", 0)
            ef = ev.get("end_frame", sf + 50)
            if sf <= self.frame_idx <= ef:
                etype = ev.get("type", "")
                if etype == "red_flag":
                    return TRACK_RED
                elif etype == "safety_car":
                    return TRACK_SC
                elif etype == "vsc":
                    return TRACK_VSC
                elif etype == "yellow_flag":
                    return TRACK_YELLOW
        return TRACK_NORMAL

    def paintEvent(self, event):
        if not self.frames or self.frame_idx >= len(self.frames):
            painter = QPainter(self)
            painter.fillRect(self.rect(), BG_DARK)
            painter.setPen(TEXT_MUTED)
            painter.drawText(self.rect(), Qt.AlignCenter,
                             "No replay data loaded")
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.fillRect(self.rect(), BG_DARK)

        scale, ox, oy = self._get_transform()
        frame_data    = self.frames[self.frame_idx]
        track_col     = self._get_track_colour()

        # ── Draw track fill ───────────────────────────────────────────────
        painter.save()
        # Build combined outer+inner polygon for track surface fill
        outer_pts = []
        for x, y in zip(self.track["x_outer"], self.track["y_outer"]):
            sx, sy = self._to_screen(x, y, scale, ox, oy)
            outer_pts.append(QPointF(sx, sy))

        inner_pts = []
        for x, y in zip(self.track["x_inner"], self.track["y_inner"]):
            sx, sy = self._to_screen(x, y, scale, ox, oy)
            inner_pts.append(QPointF(sx, sy))

        # Draw outer wall
        pen = QPen(WALL_COLOUR, max(2, scale * 3))
        pen.setCapStyle(Qt.RoundCap)
        pen.setJoinStyle(Qt.RoundJoin)
        painter.setPen(pen)
        painter.setBrush(QBrush(track_col))

        if outer_pts:
            poly_outer = QPolygonF(outer_pts)
            path_outer = QPainterPath()
            path_outer.addPolygon(poly_outer)
            path_outer.closeSubpath()
            painter.drawPath(path_outer)

        # Draw inner wall (cutout)
        if inner_pts:
            painter.setBrush(QBrush(BG_DARK))
            poly_inner = QPolygonF(inner_pts)
            path_inner = QPainterPath()
            path_inner.addPolygon(poly_inner)
            path_inner.closeSubpath()
            painter.drawPath(path_inner)

        painter.restore()

        # ── Draw DRS zones ────────────────────────────────────────────────
        if self.track.get("drs_zones"):
            pen_drs = QPen(DRS_COLOUR, max(3, scale * 5))
            pen_drs.setCapStyle(Qt.RoundCap)
            painter.setPen(pen_drs)
            for dz in self.track["drs_zones"]:
                x1, y1 = self._to_screen(
                    dz["start_x"], dz["start_y"], scale, ox, oy)
                x2, y2 = self._to_screen(
                    dz["end_x"], dz["end_y"], scale, ox, oy)
                painter.drawLine(int(x1), int(y1), int(x2), int(y2))

        # ── Draw start/finish line ────────────────────────────────────────
        if len(self.track["x_ref"]) > 0:
            sfx, sfy = self._to_screen(
                float(self.track["x_ref"][0]),
                float(self.track["y_ref"][0]),
                scale, ox, oy
            )
            pen_sf = QPen(SF_COLOUR, 2)
            painter.setPen(pen_sf)
            painter.drawLine(int(sfx) - 8, int(sfy),
                             int(sfx) + 8, int(sfy))

        # ── Draw cars ─────────────────────────────────────────────────────
        drivers_sorted = sorted(
            frame_data["drivers"].items(),
            key=lambda kv: kv[1].get("position", 20),
            reverse=True,   # draw lower positions first so P1 is on top
        )

        dot_radius = max(5, int(scale * 4))

        for code, pos in drivers_sorted:
            if "x" not in pos or "y" not in pos:
                continue

            sx, sy = self._to_screen(pos["x"], pos["y"], scale, ox, oy)

            # Team colour = border, Tyre colour = fill
            hex_team  = TEAM_COLOURS.get(
                DRIVERS_2026.get(code, (None, "Unknown"))[1], "#888888"
            )
            tyre_int  = pos.get("tyre", 0)
            fill_col  = TYRE_COLOURS_QT.get(tyre_int, QColor("#888888"))
            border_col = QColor(hex_team)

            # Draw filled circle
            painter.setBrush(QBrush(fill_col))
            pen_car = QPen(border_col, 2)
            painter.setPen(pen_car)
            painter.drawEllipse(
                QRectF(sx - dot_radius, sy - dot_radius,
                       dot_radius * 2, dot_radius * 2)
            )

            # Driver code label
            font = QFont("Arial", max(7, dot_radius - 1), QFont.Bold)
            painter.setFont(font)
            painter.setPen(QPen(TEXT_PRIMARY))
            painter.drawText(
                int(sx + dot_radius + 2),
                int(sy + 4),
                code
            )

        # ── Draw Safety Car ───────────────────────────────────────────────
        sc = frame_data.get("safety_car")
        if sc and sc.get("phase") != "returning":
            alpha = sc.get("alpha", 1.0)
            sx, sy = self._to_screen(sc["x"], sc["y"], scale, ox, oy)
            sc_col = QColor(255, 140, 0, int(alpha * 255))
            sc_r   = dot_radius + 3

            # Draw diamond shape
            pts = [
                QPointF(sx,        sy - sc_r),
                QPointF(sx + sc_r, sy),
                QPointF(sx,        sy + sc_r),
                QPointF(sx - sc_r, sy),
            ]
            painter.setBrush(QBrush(sc_col))
            painter.setPen(QPen(QColor("#ffaa00"), 2))
            painter.drawPolygon(QPolygonF(pts))

            # SC label
            painter.setPen(QPen(QColor("#ffffff")))
            font_sc = QFont("Arial", max(7, sc_r - 2), QFont.Bold)
            painter.setFont(font_sc)
            painter.drawText(int(sx + sc_r + 2), int(sy + 4), "SC")

        # ── Draw lap/time overlay ─────────────────────────────────────────
        leader_lap = max(
            (v.get("lap", 1) for v in frame_data["drivers"].values()),
            default=1
        )
        t_curr     = frame_data.get("t", 0)
        total_laps = self.data.get("total_laps", 0)
        event_name = self.data.get("event_name", "")

        painter.setPen(QPen(TEXT_PRIMARY))
        font_hud = QFont("Arial", 12, QFont.Bold)
        painter.setFont(font_hud)
        painter.drawText(10, 20,
                         f"{event_name}  |  Lap {leader_lap}/{total_laps}  |  T+{t_curr:.0f}s")

        # Track status badge
        status_text  = ""
        status_colour = QColor("#00cc44")
        for ev in self.events:
            sf = ev.get("frame", 0)
            ef = ev.get("end_frame", sf + 50)
            if sf <= self.frame_idx <= ef:
                etype = ev.get("type", "")
                if etype == "red_flag":
                    status_text  = "  RED FLAG  "
                    status_colour = QColor("#cc0000")
                elif etype == "safety_car":
                    status_text  = "  SAFETY CAR  "
                    status_colour = QColor("#ff8c00")
                elif etype == "vsc":
                    status_text  = "  VIRTUAL SC  "
                    status_colour = QColor("#ffaa00")
                elif etype == "yellow_flag":
                    status_text  = "  YELLOW  "
                    status_colour = QColor("#cccc00")

        if status_text:
            fm   = QFontMetrics(font_hud)
            tw_s = fm.horizontalAdvance(status_text)
            painter.fillRect(
                QRectF(self.width() - tw_s - 20, 4, tw_s + 10, 22),
                QBrush(status_colour)
            )
            painter.setPen(QPen(QColor("#000000")))
            painter.drawText(int(self.width() - tw_s - 15), 20, status_text)

    def mousePressEvent(self, event):
        """Click on a car dot to select it."""
        if not self.frames or self.frame_idx >= len(self.frames):
            return
        scale, ox, oy = self._get_transform()
        frame_data    = self.frames[self.frame_idx]
        dot_r         = max(5, int(scale * 4)) + 4

        mx, my = event.position().x(), event.position().y()

        for code, pos in frame_data["drivers"].items():
            if "x" not in pos:
                continue
            sx, sy = self._to_screen(pos["x"], pos["y"], scale, ox, oy)
            if abs(mx - sx) < dot_r and abs(my - sy) < dot_r:
                self.car_clicked.emit(code)
                return


# ── Main replay window ─────────────────────────────────────────────────────
class ReplayWindow(QMainWindow):

    def __init__(self, year, round_number, session_type="R"):
        super().__init__()
        self.year         = year
        self.round_number = round_number
        self.session_type = session_type
        self.data         = None
        self.frame_idx    = 0
        self.playing      = False
        self.speed        = 1.0
        self.selected_drv = None

        self.setWindowTitle(f"F1 {year} Race Replay")
        self.setMinimumSize(1100, 700)
        self.setStyleSheet("background-color: #0d0d0d; color: #f0f0f0;")

        self._build_ui()
        self._start_timer()
        self._load_data()

    # ── UI construction ───────────────────────────────────────────────────
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        # Left: track canvas + controls
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(6)

        self.canvas = TrackCanvas()
        self.canvas.car_clicked.connect(self._on_car_clicked)
        left_layout.addWidget(self.canvas, stretch=1)

        # Controls bar
        ctrl_widget = QWidget()
        ctrl_widget.setStyleSheet("background-color: #111; border-radius: 6px;")
        ctrl_layout = QHBoxLayout(ctrl_widget)
        ctrl_layout.setContentsMargins(10, 6, 10, 6)
        ctrl_layout.setSpacing(10)

        self.btn_play = QPushButton("  Play")
        self.btn_play.setFixedWidth(80)
        self.btn_play.setStyleSheet(self._btn_style("#226622"))
        self.btn_play.clicked.connect(self._toggle_play)

        btn_restart = QPushButton("  Restart")
        btn_restart.setFixedWidth(80)
        btn_restart.setStyleSheet(self._btn_style("#333"))
        btn_restart.clicked.connect(self._restart)

        self.speed_combo = QComboBox()
        self.speed_combo.addItems(["0.25x", "0.5x", "1x", "2x", "4x"])
        self.speed_combo.setCurrentIndex(2)
        self.speed_combo.setFixedWidth(70)
        self.speed_combo.setStyleSheet(
            "background:#222; color:#f0f0f0; border:1px solid #444; padding:2px;")
        self.speed_combo.currentTextChanged.connect(self._on_speed_change)

        self.lbl_frame = QLabel("Frame 0 / 0")
        self.lbl_frame.setStyleSheet("color: #888; font-size: 11px;")

        # Scrubber
        self.scrubber = QSlider(Qt.Horizontal)
        self.scrubber.setMinimum(0)
        self.scrubber.setValue(0)
        self.scrubber.setStyleSheet("""
            QSlider::groove:horizontal { height:6px; background:#333; border-radius:3px; }
            QSlider::handle:horizontal { width:14px; height:14px; margin:-4px 0;
                background:#e8002d; border-radius:7px; }
            QSlider::sub-page:horizontal { background:#e8002d; border-radius:3px; }
        """)
        self.scrubber.sliderMoved.connect(self._on_scrub)

        ctrl_layout.addWidget(self.btn_play)
        ctrl_layout.addWidget(btn_restart)
        ctrl_layout.addWidget(QLabel("Speed:"))
        ctrl_layout.addWidget(self.speed_combo)
        ctrl_layout.addWidget(self.scrubber, stretch=1)
        ctrl_layout.addWidget(self.lbl_frame)

        left_layout.addWidget(ctrl_widget)
        main_layout.addWidget(left_panel, stretch=3)

        # Right: leaderboard + driver info
        right_panel = QFrame()
        right_panel.setFrameShape(QFrame.StyledPanel)
        right_panel.setFixedWidth(260)
        right_panel.setStyleSheet(
            "background-color: #111111; border-radius: 8px; border: 1px solid #222;")
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(8, 8, 8, 8)
        right_layout.setSpacing(6)

        lbl_lb = QLabel("LEADERBOARD")
        lbl_lb.setStyleSheet(
            "color: #888; font-size: 10px; font-weight: bold; letter-spacing: 2px;")
        right_layout.addWidget(lbl_lb)

        self.leaderboard = QTableWidget(22, 4)
        self.leaderboard.setHorizontalHeaderLabels(["Pos", "Driver", "Lap", "Tyre"])
        self.leaderboard.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.leaderboard.verticalHeader().setVisible(False)
        self.leaderboard.setSelectionBehavior(QTableWidget.SelectRows)
        self.leaderboard.setEditTriggers(QTableWidget.NoEditTriggers)
        self.leaderboard.setStyleSheet("""
            QTableWidget { background:#0d0d0d; color:#f0f0f0;
                           gridline-color:#222; font-size:12px; border:none; }
            QTableWidget::item { padding:3px; }
            QTableWidget::item:selected { background:#2a2a2a; }
            QHeaderView::section { background:#1a1a1a; color:#888;
                                   font-size:10px; border:none; padding:4px; }
        """)
        right_layout.addWidget(self.leaderboard, stretch=1)

        # Selected driver info panel
        self.drv_info = QLabel("Click a car on the track\nto see telemetry")
        self.drv_info.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.drv_info.setStyleSheet(
            "color:#aaa; font-size:12px; padding:8px;"
            "background:#0d0d0d; border-radius:6px; border:1px solid #222;")
        self.drv_info.setFixedHeight(140)
        right_layout.addWidget(self.drv_info)

        main_layout.addWidget(right_panel, stretch=0)

    def _btn_style(self, bg):
        return (f"QPushButton {{ background:{bg}; color:#f0f0f0; "
                f"border:1px solid #444; border-radius:4px; padding:5px 8px; "
                f"font-size:12px; }}"
                f"QPushButton:hover {{ background:#444; }}"
                f"QPushButton:pressed {{ background:#222; }}")

    # ── Data loading ──────────────────────────────────────────────────────
    def _load_data(self):
        self.setWindowTitle(
            f"F1 {self.year} R{self.round_number} -- Loading...")

        self.loader = DataLoaderThread(
            self.year, self.round_number, self.session_type)
        self.loader.finished.connect(self._on_data_loaded)
        self.loader.error.connect(self._on_load_error)
        self.loader.progress.connect(
            lambda msg: self.setWindowTitle(msg))
        self.loader.start()

    def _on_data_loaded(self, data):
        self.data      = data
        circuit_name   = data.get("event_name", f"Round {self.round_number}")
        self.setWindowTitle(
            f"F1 {self.year} -- {circuit_name} -- Race Replay")
        self.canvas.set_data(data)
        self.scrubber.setMaximum(len(data["frames"]) - 1)
        self._update_leaderboard()

    def _on_load_error(self, msg):
        self.setWindowTitle(f"Error: {msg}")

    # ── Timer / playback ──────────────────────────────────────────────────
    def _start_timer(self):
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)

    def _tick(self):
        if self.data is None:
            return
        n = len(self.data["frames"])
        if self.frame_idx < n - 1:
            self.frame_idx += 1
            self.canvas.set_frame(self.frame_idx)
            self.scrubber.setValue(self.frame_idx)
            self.lbl_frame.setText(f"Frame {self.frame_idx} / {n - 1}")
            self._update_leaderboard()
            if self.selected_drv:
                self._update_drv_info(self.selected_drv)
        else:
            self._pause()

    def _toggle_play(self):
        if self.playing:
            self._pause()
        else:
            self._play()

    def _play(self):
        if self.data is None:
            return
        self.playing = True
        self.btn_play.setText("  Pause")
        self.btn_play.setStyleSheet(self._btn_style("#662222"))
        interval = max(16, int(100 / self.speed))
        self.timer.start(interval)

    def _pause(self):
        self.playing = False
        self.btn_play.setText("  Play")
        self.btn_play.setStyleSheet(self._btn_style("#226622"))
        self.timer.stop()

    def _restart(self):
        self._pause()
        self.frame_idx = 0
        if self.data:
            self.canvas.set_frame(0)
            self.scrubber.setValue(0)
            self._update_leaderboard()

    def _on_scrub(self, value):
        self.frame_idx = value
        self.canvas.set_frame(value)
        if self.data:
            self._update_leaderboard()

    def _on_speed_change(self, text):
        self.speed = float(text.replace("x", ""))
        if self.playing:
            interval = max(16, int(100 / self.speed))
            self.timer.setInterval(interval)

    # ── Leaderboard update ────────────────────────────────────────────────
    def _update_leaderboard(self):
        if not self.data or self.frame_idx >= len(self.data["frames"]):
            return

        frame_data     = self.data["frames"][self.frame_idx]
        driver_colours = self.data.get("driver_colours", {})
        drivers_sorted = sorted(
            frame_data["drivers"].items(),
            key=lambda kv: kv[1].get("position", 99)
        )

        self.leaderboard.setRowCount(len(drivers_sorted))

        for row, (code, pos) in enumerate(drivers_sorted):
            info     = DRIVERS_2026.get(code, (code, "Unknown", ""))
            team     = info[1]
            pos_num  = pos.get("position", row + 1)
            lap      = pos.get("lap", 1)
            tyre_int = pos.get("tyre", 0)
            tyre_str = TYRE_NAMES.get(tyre_int, "?")

            hex_team = TEAM_COLOURS.get(team, "#888888")
            team_col = QColor(hex_team)

            items = [
                QTableWidgetItem(str(pos_num)),
                QTableWidgetItem(code),
                QTableWidgetItem(str(lap)),
                QTableWidgetItem(tyre_str),
            ]

            for col, item in enumerate(items):
                item.setForeground(QBrush(team_col))
                item.setTextAlignment(Qt.AlignCenter)
                self.leaderboard.setItem(row, col, item)

    # ── Driver info panel ─────────────────────────────────────────────────
    def _on_car_clicked(self, code):
        self.selected_drv = code
        self._update_drv_info(code)

    def _update_drv_info(self, code):
        if not self.data or self.frame_idx >= len(self.data["frames"]):
            return

        frame_data = self.data["frames"][self.frame_idx]
        pos        = frame_data["drivers"].get(code)
        if not pos:
            return

        info     = DRIVERS_2026.get(code, (code, "Unknown", ""))
        team     = info[1]
        fullname = info[0]
        pos_num  = pos.get("position", "?")
        speed    = pos.get("speed", 0)
        gear     = pos.get("gear", 0)
        drs_on   = int(pos.get("drs", 0)) >= 10
        lap      = pos.get("lap", 1)
        tyre_str = TYRE_NAMES.get(pos.get("tyre", 0), "?")

        hex_team = TEAM_COLOURS.get(team, "#888888")

        self.drv_info.setText(
            f"<span style='color:{hex_team}; font-weight:bold; font-size:14px;'>"
            f"P{pos_num}  {fullname}</span><br>"
            f"<span style='color:#888;'>{team}</span><br><br>"
            f"Lap: <b>{lap}</b><br>"
            f"Speed: <b>{speed:.0f} km/h</b><br>"
            f"Gear: <b>{gear}</b> &nbsp;&nbsp; "
            f"DRS: <b style='color:{'#00cc44' if drs_on else '#888'};'>"
            f"{'ON' if drs_on else 'OFF'}</b><br>"
            f"Tyre: <b style='color:{TYRE_COLOURS_QT.get(pos.get('tyre',0), QColor('#888')).name()};'>"
            f"{tyre_str}</b>"
        )
        self.drv_info.setTextFormat(Qt.RichText)


# ── Entry point ───────────────────────────────────────────────────────────
def launch_replay(year=2026, round_number=1, session_type="R"):
    """Launch the replay window. Can be called from Streamlit or CLI."""
    app = QApplication.instance() or QApplication(sys.argv)
    window = ReplayWindow(year, round_number, session_type)
    window.show()
    app.exec()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="F1 Race Replay")
    parser.add_argument("--year",    type=int, default=2026)
    parser.add_argument("--round",   type=int, default=1)
    parser.add_argument("--session", type=str, default="R",
                        help="R=Race, Q=Qualifying")
    args = parser.parse_args()
    launch_replay(args.year, args.round, args.session)