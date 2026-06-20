"""Central configuration for TrafficPulse."""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
MODELS_DIR = PROJECT_DIR / "models"
SRC_DIR = PROJECT_DIR / "src"

INPUT_CSV = DATA_DIR / "Astram event data_anonymized - Astram event data_anonymizedb40ac87.csv"
FEATURES_CSV = DATA_DIR / "features_all_8173.csv"
EDA_REPORT = DATA_DIR / "eda_summary.txt"

CLOSURE_CLASSIFIER_MODEL_PATH = MODELS_DIR / "closure_classifier.pkl"
SEVERITY_MODEL_PATH = MODELS_DIR / "severity_regressor.pkl"
DURATION_MODEL_PATH = MODELS_DIR / "duration_lookup.pkl"
ROAD_GRAPH_PATH = MODELS_DIR / "road_graph.pkl"
FEATURE_NAMES_PATH = MODELS_DIR / "feature_names.pkl"

# ── Feature Engineering & EDA Constants ──────────────────────────────
PEAK_HOURS = [8, 9, 10, 17, 18, 19]
WEEKEND_DAYS = [5, 6]
EXPECTED_ROWS = 8173
EXPECTED_ORIGINAL_COLUMNS = 46
EXPECTED_OUTPUT_COLUMNS = 57

# ── App & Map Configuration ───────────────────────────────────────────
BLR_LAT, BLR_LON = 12.9716, 77.5946

RISK_LOW_THRESHOLD = 0.35
RISK_HIGH_THRESHOLD = 0.65

MAP_GREEN_THRESHOLD = 0.45
MAP_AMBER_THRESHOLD = 0.72

MIN_RADIUS_KM = 0.5
MAX_RADIUS_KM = 5.0

MIN_PATROL_UNITS = 3
PATROL_MULTIPLIER = 15
MIN_BARRICADE_POINTS = 2
BARRICADE_MULTIPLIER = 6

DEFAULT_DIVERSION_ROUTE = "Mysore Inner Ring Road"

# ── Corridor → Real-World Coordinates (FIX for location/route bug) ───
# ROOT CAUSE OF THE BUG: the app previously derived map coordinates by
# geohash2-decoding the synthetic graph node IDs (e.g. "tumh"), which
# decode to a region of Bihar, India (~24N, 86E) and were then shifted
# by a single global lat/lon offset. That produces a cluster of points
# near Shivamogga/Chikkamagaluru (~13.5N, 75.6E) — nowhere near
# Bengaluru — which is exactly what was visible on the broken maps.
#
# Fix: map each real ASTraM `corridor` value (these are the literal
# one-hot columns the trained model expects — see models/feature_names.pkl)
# directly to an actual Bengaluru-area lat/lon. These are corridor-level
# anchor points (a corridor is a road stretch, not a single GPS pin), not
# survey-precise coordinates, but they are real, verifiable locations in
# the correct part of the city — CBD coordinates cross-checked against
# Vidhana Soudha / MG Road public records.
CORRIDOR_COORDINATES = {
    "Mysore Road":             (12.9447, 77.5301),  # Nayandahalli / RR Nagar, SW (NH275)
    "Bellary Road 1":          (13.0358, 77.5970),  # Hebbal, N (NH44)
    "Bellary Road 2":          (13.0850, 77.5950),  # Yelahanka stretch, further N
    "Hosur Road":              (12.9165, 77.6224),  # Madiwala / BTM, S (NH44)
    "Tumkur Road":             (13.0210, 77.5310),  # Peenya, NW (NH4)
    "Old Madras Road":         (12.9950, 77.6680),  # Towards KR Puram, NE
    "ORR North 1":             (13.0450, 77.6150),  # Hebbal–Nagawara stretch
    "ORR North 2":             (13.0300, 77.6450),  # Nagawara–KR Puram stretch
    "ORR East 1":              (12.9590, 77.6970),  # Marathahalli stretch
    "ORR East 2":              (12.9270, 77.6920),  # Marathahalli–Sarjapur stretch
    "ORR West 1":              (12.9230, 77.5150),  # Nayandahalli–RR Nagar stretch
    "Bannerghata Road":        (12.8990, 77.5970),  # South Bengaluru
    "Hennur Main Road":        (13.0370, 77.6390),  # North-East Bengaluru
    "IRR(Thanisandra road)":   (13.0620, 77.6190),  # North Bengaluru
    "Magadi Road":             (12.9770, 77.5440),  # West Bengaluru
    "Old Airport Road":        (12.9610, 77.6620),  # HAL / Old Airport area, East
    "Airport New South Road":  (13.1700, 77.6850),  # South side of KIA Airport
    "Varthur Road":            (12.9420, 77.7380),  # Whitefield / Varthur, East
    "West of Chord Road":      (12.9920, 77.5530),  # Rajajinagar area
    "CBD 1":                   (12.9755, 77.6068),  # MG Road / Vidhana Soudha (verified)
    "CBD 2":                   (12.9700, 77.6090),  # Brigade Rd / Commercial St
}

# Catch-all categories present in the training data that are NOT real,
# selectable map locations — excluded from the UI dropdown deliberately.
NON_SELECTABLE_CORRIDORS = {"Non-corridor", "Unknown"}

# Each corridor needs a graph node (for the spillover/diversion simulation
# math only — NOT for plotting). These are arbitrary-but-unique existing
# nodes in models/road_graph.pkl; the simulation only cares about graph
# topology (who is whose neighbour), not what the node ID literally decodes to.
CORRIDOR_GEOHASH_MAP = {
    "Mysore Road":             "tumh",
    "Bellary Road 1":          "tumj",
    "Bellary Road 2":          "tumk",
    "Hosur Road":              "tumc",
    "Tumkur Road":             "tumf",
    "Old Madras Road":         "tumm",
    "ORR North 1":             "tums",
    "ORR East 1":              "tumg",
    "Bannerghata Road":        "tukn",
    "Hennur Main Road":        "tukq",
    "IRR(Thanisandra road)":   "tukr",
    "Magadi Road":             "tuks",
    "Old Airport Road":        "tukt",
    "Airport New South Road":  "tume",
    "Varthur Road":            "tumv",
    "West of Chord Road":      "tumw",
    "CBD 1":                   "tumx",
    "CBD 2":                   "tumy",
    "ORR North 2":             "tumz",
    "ORR East 2":              "tupb",
    "ORR West 1":              "tupc",
}

# ── Simulation Engine Constants ───────────────────────────────────────
TIME_AMPLITUDE = 0.3
PEAK_OFFSET_HOURS = 8

DIRECT_IMPACT_MULTIPLIER = 2.5
ONE_HOP_SPILLOVER = 0.3
TWO_HOP_SPILLOVER = 0.1

DIVERSION_RELIEF_FACTOR = 0.6
DIVERSION_LOAD_FACTOR = 1.15

# ── Pre-Defined Diversion Routes (metadata for Bengaluru corridors) ───
# The actual diversion arc waypoints are computed dynamically in app.py
# relative to the event location. This config stores per-corridor metadata:
#   route_name  – human-readable alternate route label
#   bypass_side – "west" or "east" — which side of the corridor to arc around
#   improvement – expected congestion reduction fraction
#   description – textual description of the alternate route
DIVERSION_ROUTES = {
    "Mysore Road":             {"route_name": "Via Inner Ring Road & RR Nagar",        "bypass_side": "west",  "improvement": 0.22, "description": "Bypasses Mysore Road via IRR & RR Nagar"},
    "Bellary Road 1":          {"route_name": "Via ORR North (Hebbal Bypass)",         "bypass_side": "west",  "improvement": 0.25, "description": "Bypasses Bellary Road via ORR North & Yeshwanthpur"},
    "Bellary Road 2":          {"route_name": "Via Yelahanka Bypass",                  "bypass_side": "west",  "improvement": 0.20, "description": "Bypasses Yelahanka stretch via Vidyaranyapura"},
    "Hosur Road":              {"route_name": "Via Bannerghatta Road & BTM",           "bypass_side": "west",  "improvement": 0.24, "description": "Bypasses Hosur Road via BTM & Bannerghatta Road"},
    "Tumkur Road":             {"route_name": "Via Peenya Industrial Bypass",          "bypass_side": "west",  "improvement": 0.21, "description": "Bypasses Tumkur Road via Peenya Industrial Area"},
    "Old Madras Road":         {"route_name": "Via Indiranagar & CV Raman Nagar",      "bypass_side": "east",  "improvement": 0.23, "description": "Bypasses Old Madras Road via Indiranagar"},
    "ORR North 1":             {"route_name": "Via Hennur & Thanisandra",              "bypass_side": "east",  "improvement": 0.19, "description": "Bypasses ORR North via Hennur & Thanisandra"},
    "ORR North 2":             {"route_name": "Via Kalyan Nagar & HRBR Layout",        "bypass_side": "west",  "improvement": 0.18, "description": "Bypasses ORR North 2 via Kalyan Nagar"},
    "ORR East 1":              {"route_name": "Via Whitefield & Kundalahalli",         "bypass_side": "east",  "improvement": 0.20, "description": "Bypasses ORR East via Whitefield & Kundalahalli"},
    "ORR East 2":              {"route_name": "Via Sarjapur Road Inner",               "bypass_side": "east",  "improvement": 0.22, "description": "Bypasses ORR East 2 via Sarjapur Road"},
    "ORR West 1":              {"route_name": "Via Kengeri & Uttarahalli",             "bypass_side": "west",  "improvement": 0.21, "description": "Bypasses ORR West via Kengeri & Uttarahalli"},
    "Bannerghata Road":        {"route_name": "Via JP Nagar & Jayanagar",              "bypass_side": "west",  "improvement": 0.23, "description": "Bypasses Bannerghatta Road via JP Nagar"},
    "Hennur Main Road":        {"route_name": "Via Kalyan Nagar & Banaswadi",          "bypass_side": "west",  "improvement": 0.20, "description": "Bypasses Hennur Main Road via Kalyan Nagar"},
    "IRR(Thanisandra road)":   {"route_name": "Via Jakkur & Yelahanka",                "bypass_side": "west",  "improvement": 0.19, "description": "Bypasses Thanisandra Road via Jakkur & Yelahanka"},
    "Magadi Road":             {"route_name": "Via Chord Road & Rajajinagar",          "bypass_side": "west",  "improvement": 0.22, "description": "Bypasses Magadi Road via Chord Road & Rajajinagar"},
    "Old Airport Road":        {"route_name": "Via Indiranagar & Domlur",              "bypass_side": "west",  "improvement": 0.24, "description": "Bypasses Old Airport Road via Indiranagar & Domlur"},
    "Airport New South Road":  {"route_name": "Via Devanahalli Bypass",                "bypass_side": "west",  "improvement": 0.18, "description": "Bypasses Airport South Road via Devanahalli"},
    "Varthur Road":            {"route_name": "Via ITPL & Whitefield",                 "bypass_side": "west",  "improvement": 0.20, "description": "Bypasses Varthur Road via ITPL & Whitefield"},
    "West of Chord Road":      {"route_name": "Via Rajajinagar & Basaveshwaranagar",   "bypass_side": "west",  "improvement": 0.21, "description": "Bypasses West of Chord Road via Basaveshwaranagar"},
    "CBD 1":                   {"route_name": "Via Shivajinagar & Frazer Town",        "bypass_side": "west",  "improvement": 0.26, "description": "Bypasses CBD via Shivajinagar & Frazer Town"},
    "CBD 2":                   {"route_name": "Via Richmond Road & Langford",          "bypass_side": "west",  "improvement": 0.25, "description": "Bypasses CBD 2 via Richmond Road & Langford"},
}

