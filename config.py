"""
config.py -- Central config for the 2026 F1 Prediction System.
All 2026 driver/team data, known results, and model settings live here.
Update RESULTS_2026 after each race weekend.
"""

import os

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, "data", "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# ── Training seasons (FastF1 pulls these) ──────────────────────────────────
TRAIN_SEASONS = [2023, 2024]

# ── 2026 complete driver-team grid ────────────────────────────────────────
DRIVERS_2026 = {
    # code : (Full Name,           Team,           Power Unit)
    "RUS": ("George Russell",     "Mercedes",      "Mercedes"),
    "ANT": ("Kimi Antonelli",     "Mercedes",      "Mercedes"),
    "LEC": ("Charles Leclerc",    "Ferrari",       "Ferrari"),
    "HAM": ("Lewis Hamilton",     "Ferrari",       "Ferrari"),
    "NOR": ("Lando Norris",       "McLaren",       "Mercedes"),
    "PIA": ("Oscar Piastri",      "McLaren",       "Mercedes"),
    "VER": ("Max Verstappen",     "Red Bull",      "Red Bull"),
    "HAD": ("Isack Hadjar",       "Red Bull",      "Red Bull"),
    "ALO": ("Fernando Alonso",    "Aston Martin",  "Honda"),
    "STR": ("Lance Stroll",       "Aston Martin",  "Honda"),
    "ALB": ("Alex Albon",         "Williams",      "Mercedes"),
    "SAI": ("Carlos Sainz",       "Williams",      "Mercedes"),
    "GAS": ("Pierre Gasly",       "Alpine",        "Renault"),
    "COL": ("Franco Colapinto",   "Alpine",        "Renault"),
    "LAW": ("Liam Lawson",        "Racing Bulls",  "Red Bull"),
    "LIN": ("Arvid Lindblad",     "Racing Bulls",  "Red Bull"),
    "BEA": ("Oliver Bearman",     "Haas",          "Ferrari"),
    "OCO": ("Esteban Ocon",       "Haas",          "Ferrari"),
    "HUL": ("Nico Hulkenberg",    "Audi",          "Audi"),
    "BOR": ("Gabriel Bortoleto",  "Audi",          "Audi"),
    "PER": ("Sergio Perez",       "Cadillac",      "Ferrari"),
    "BOT": ("Valtteri Bottas",    "Cadillac",      "Ferrari"),
}

# ── 2026 car numbers (changed from 2025) ─────────────────────────────────
# Key changes: Verstappen dropped #1 (now #3), Norris took #1, Hamilton #44
# FastF1 uses car numbers internally -- this map converts number -> driver code
CAR_NUMBERS_2026 = {
    63:  "RUS",   # Russell
    12:  "ANT",   # Antonelli
    16:  "LEC",   # Leclerc
    44:  "HAM",   # Hamilton (permanent #44)
    1:   "NOR",   # Norris (took #1 from Verstappen)
    81:  "PIA",   # Piastri
    3:   "VER",   # Verstappen (dropped #1, back to #3)
    6:   "HAD",   # Hadjar
    14:  "ALO",   # Alonso
    18:  "STR",   # Stroll
    23:  "ALB",   # Albon
    55:  "SAI",   # Sainz
    10:  "GAS",   # Gasly
    43:  "COL",   # Colapinto
    30:  "LAW",   # Lawson
    87:  "LIN",   # Lindblad (rookie, new number)
    5:   "BEA",   # Bearman
    31:  "OCO",   # Ocon
    27:  "HUL",   # Hulkenberg
    77:  "BOT",   # Bottas -- wait Bottas is 77 -- actually check
    11:  "BOR",   # Bortoleto
    2:   "PER",   # Perez
}

# Reverse map: driver code -> car number
DRIVER_NUMBERS_2026 = {v: k for k, v in CAR_NUMBERS_2026.items()}

# ── F1 points system ──────────────────────────────────────────────────────
POINTS = {1:25, 2:18, 3:15, 4:12, 5:10, 6:8, 7:6, 8:4, 9:2, 10:1}

# Sprint points system (top 8 only: 8-7-6-5-4-3-2-1)
SPRINT_POINTS = {1:8, 2:7, 3:6, 4:5, 5:4, 6:3, 7:2, 8:1}

# ── 2026 known results ────────────────────────────────────────────────────
# RESULTS_2026: main race results only {round: {driver: position}}
# SPRINT_RESULTS_2026: sprint results {round: {driver: position}}
# DNF=21, DNS=22

SPRINT_RESULTS_2026 = {
    2: {  # China Sprint -- March 22
        "RUS": 1,
        "LEC": 2,
        "HAM": 3,
        "NOR": 4,
        "ANT": 5,
        "PIA": 6,
        "LAW": 7,
        "BEA": 8,
        "VER": 9,
        "OCO": 10,
        "GAS": 11,
        "SAI": 12,
        "BOR": 13,
        "COL": 14,
        "HAD": 15,
        "ALB": 16,
        "ALO": 17,
        "STR": 18,
        "PER": 19,
        "HUL": 21,   # DNF
        "BOT": 21,   # DNF
        "LIN": 21,   # DNF (opening lap spin)
    },
}

RESULTS_2026 = {
    1: {  # Australia -- March 16
        "RUS": 1,
        "ANT": 2,
        "LEC": 3,
        "HAM": 4,
        "NOR": 5,
        "VER": 6,
        "BEA": 7,
        "LIN": 8,
        "BOR": 9,
        "GAS": 10,
        "HUL": 11,
        "LAW": 12,
        "SAI": 13,
        "ALB": 14,
        "STR": 15,
        "COL": 16,
        "BOT": 17,
        "HAD": 18,
        "OCO": 19,
        "ALO": 21,   # DNF
        "PIA": 21,   # DNF
        "PER": 22,   # DNS
    },
    2: {  # China -- March 23
        "ANT": 1,
        "RUS": 2,
        "HAM": 3,
        "LEC": 4,
        "BEA": 5,
        "GAS": 6,
        "LAW": 7,
        "HAD": 8,
        "SAI": 9,
        "COL": 10,
        "HUL": 11,
        "LIN": 12,
        "BOT": 13,
        "OCO": 14,
        "PER": 15,
        "VER": 21,   # DNF
        "ALO": 21,   # DNF
        "STR": 21,   # DNF
        "NOR": 22,   # DNS
        "PIA": 22,   # DNS
        "BOR": 22,   # DNS
        "ALB": 22,   # DNS
    },
}

# ── 2026 calendar (22 races) ──────────────────────────────────────────────
CALENDAR_2026 = {
    1:  ("Australia",    "Albert Park",           "2026-03-16"),
    2:  ("China",        "Shanghai",              "2026-03-23"),
    3:  ("Japan",        "Suzuka",                "2026-03-29"),
    4:  ("Bahrain",      "Bahrain International", "2026-04-13"),
    5:  ("Saudi Arabia", "Jeddah",                "2026-04-20"),
    6:  ("Miami",        "Miami",                 "2026-05-04"),
    7:  ("Emilia Romagna","Imola",                "2026-05-18"),
    8:  ("Monaco",       "Monaco",                "2026-05-25"),
    9:  ("Spain",        "Barcelona",             "2026-06-01"),
    10: ("Canada",       "Montreal",              "2026-06-15"),
    11: ("Austria",      "Red Bull Ring",         "2026-06-29"),
    12: ("Britain",      "Silverstone",           "2026-07-06"),
    13: ("Belgium",      "Spa",                   "2026-07-27"),
    14: ("Hungary",      "Hungaroring",           "2026-08-03"),
    15: ("Netherlands",  "Zandvoort",             "2026-08-31"),
    16: ("Italy",        "Monza",                 "2026-09-07"),
    17: ("Azerbaijan",   "Baku",                  "2026-09-21"),
    18: ("Singapore",    "Marina Bay",            "2026-10-05"),
    19: ("United States","COTA",                  "2026-10-19"),
    20: ("Mexico",       "Mexico City",           "2026-10-26"),
    21: ("Brazil",       "Interlagos",            "2026-11-09"),
    22: ("Abu Dhabi",    "Yas Marina",            "2026-11-29"),
}

# ── Circuit type tags (affects strategy/pace model) ───────────────────────
CIRCUIT_TYPES = {
    "Albert Park":           "street_hybrid",
    "Shanghai":              "high_downforce",
    "Suzuka":                "high_speed",
    "Bahrain International": "mixed",
    "Jeddah":                "street_fast",
    "Miami":                 "street_hybrid",
    "Imola":                 "technical",
    "Monaco":                "street_slow",
    "Barcelona":             "mixed",
    "Montreal":              "stop_go",
    "Red Bull Ring":         "high_speed",
    "Silverstone":           "high_speed",
    "Spa":                   "high_speed",
    "Hungaroring":           "high_downforce",
    "Zandvoort":             "high_downforce",
    "Monza":                 "low_downforce",
    "Baku":                  "street_fast",
    "Marina Bay":            "street_slow",
    "COTA":                  "mixed",
    "Mexico City":           "high_altitude",
    "Interlagos":            "mixed",
    "Yas Marina":            "mixed",
}

# ── 2026 regulation era weight ─────────────────────────────────────────────
# Multiplier applied to each season when building features.
# Higher = more influence on predictions.
# As more 2026 races complete, this auto-adjusts in the model.
ERA_WEIGHTS = {
    2023: 0.4,
    2024: 0.6,
    2026: 2.5,   # 2026 data is most relevant (new regs)
}

# ── Team colours for plots ────────────────────────────────────────────────
TEAM_COLOURS = {
    "Mercedes":     "#00D2BE",
    "Ferrari":      "#E8002D",
    "McLaren":      "#FF8000",
    "Red Bull":     "#3671C6",
    "Aston Martin": "#229971",
    "Williams":     "#64C4FF",
    "Alpine":       "#FF87BC",
    "Racing Bulls": "#6692FF",
    "Haas":         "#B6BABD",
    "Audi":         "#C0C0C0",
    "Cadillac":     "#CF9B00",
}

DRIVER_COLOURS = {
    drv: TEAM_COLOURS.get(info[1], "#888888")
    for drv, info in DRIVERS_2026.items()
}