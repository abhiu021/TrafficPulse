"""
app.py  –  Bengaluru Event-Driven Congestion Forecaster
Streamlit dashboard with two modes:
  Mode A – Planned Event Forecast
  Mode B – Live Incident Assessment
"""

from __future__ import annotations

import math
import pickle
import sys
from copy import deepcopy
from datetime import date, datetime, time, timedelta
from pathlib import Path

import folium
import networkx as nx
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

# ── Project paths ──────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))

from src.models.simulation_engine import (
    get_live_state,
    simulate_event_impact,
    simulate_with_diversion,
)
from src.config import (
    CLOSURE_CLASSIFIER_MODEL_PATH, SEVERITY_MODEL_PATH, DURATION_MODEL_PATH,
    ROAD_GRAPH_PATH, FEATURE_NAMES_PATH, BLR_LAT, BLR_LON,
    RISK_LOW_THRESHOLD, RISK_HIGH_THRESHOLD, MAP_GREEN_THRESHOLD,
    MAP_AMBER_THRESHOLD, MIN_RADIUS_KM, MAX_RADIUS_KM, MIN_PATROL_UNITS,
    PATROL_MULTIPLIER, MIN_BARRICADE_POINTS, BARRICADE_MULTIPLIER,
    DEFAULT_DIVERSION_ROUTE, CORRIDOR_COORDINATES, CORRIDOR_GEOHASH_MAP,
    DIVERSION_ROUTES
)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    layout="wide",
    page_title="Event Impact Forecaster",
    page_icon="🚔",
    initial_sidebar_state="expanded",
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown('<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap" rel="stylesheet">', unsafe_allow_html=True)
st.markdown("""
<style>
/* ── palette ── */
:root {
    --bg:       #f8fafc;
    --bg2:      #f1f5f9;
    --card:     rgba(255,255,255,0.95);
    --card2:    rgba(248,250,252,0.95);
    --card-solid: #ffffff;
    --border:   rgba(99,102,241,0.2);
    --accent:   #6366f1;
    --accent2:  #ec4899;
    --accent-g: linear-gradient(135deg, #6366f1 0%, #ec4899 100%);
    --green:    #10b981;
    --amber:    #f59e0b;
    --red:      #ef4444;
    --txt:      #0f172a;
    --sub:      #64748b;
    --glass:    rgba(0,0,0,0.03);
    --radius:   16px;
}

/* ── base ── */
html, body, [class*="css"], [class*="st-"] { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }
[data-testid="stIconMaterial"], .material-symbols-rounded, .material-icons, i { font-family: 'Material Symbols Rounded', 'Material Icons' !important; }
#MainMenu, footer { visibility: hidden; }

.stApp {
    background: var(--bg) !important;
    background-image:
        radial-gradient(ellipse 80% 50% at 50% -20%, rgba(124,106,255,0.08), transparent),
        radial-gradient(ellipse 60% 40% at 80% 100%, rgba(255,107,138,0.05), transparent) !important;
}

/* ── scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(124,106,255,0.25); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(124,106,255,0.45); }

/* ── fade-in animation ── */
@keyframes fadeSlideIn {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}
.stMainBlockContainer > div > div { animation: fadeSlideIn 0.5s ease-out both; }

/* ── sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%) !important;
    border-right: 1px solid rgba(99,102,241,0.15) !important;
}
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {
    font-size: .95rem !important;
    font-weight: 600 !important;
    color: var(--sub) !important;
    letter-spacing: 0.03em !important;
    text-transform: uppercase !important;
    margin-top: 1.2rem !important;
}
section[data-testid="stSidebar"] .stSelectbox > div > div,
section[data-testid="stSidebar"] .stDateInput > div > div,
section[data-testid="stSidebar"] .stTimeInput > div > div {
    background: rgba(255,255,255,0.9) !important;
    border: 1px solid rgba(99,102,241,0.2) !important;
    border-radius: 10px !important;
}
section[data-testid="stSidebar"] .stSlider > div > div > div {
    color: var(--accent) !important;
}

/* ── sidebar buttons ── */
section[data-testid="stSidebar"] .stButton > button {
    background: var(--accent-g) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.7rem 1.4rem !important;
    font-weight: 700 !important;
    font-size: 0.92rem !important;
    letter-spacing: 0.04em !important;
    text-transform: uppercase !important;
    box-shadow: 0 4px 20px rgba(124,106,255,0.3) !important;
    transition: all 0.25s cubic-bezier(.4,0,.2,1) !important;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(124,106,255,0.45) !important;
}
section[data-testid="stSidebar"] .stButton > button:active {
    transform: translateY(0) !important;
}

/* ── main area buttons (Test Alternate Route etc.) ── */
.stMainBlockContainer .stButton > button[kind="primary"],
.stMainBlockContainer .stButton > button[data-testid] {
    background: var(--accent-g) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.65rem 2rem !important;
    font-weight: 700 !important;
    font-size: 0.88rem !important;
    letter-spacing: 0.04em !important;
    box-shadow: 0 4px 20px rgba(124,106,255,0.25) !important;
    transition: all 0.25s cubic-bezier(.4,0,.2,1) !important;
}
.stMainBlockContainer .stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(124,106,255,0.4) !important;
}

/* ── typography ── */
h1 {
    background: var(--accent-g);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    font-size: 2.1rem !important; font-weight: 900 !important;
    letter-spacing: -0.02em !important;
    margin-bottom: 0 !important;
}
h3 {
    color: var(--sub) !important; font-weight: 400 !important;
    font-size: 1rem !important; letter-spacing: 0.01em !important;
}

/* ── subheaders with accent bar ── */
.stMainBlockContainer h2 {
    color: var(--txt) !important;
    font-weight: 700 !important;
    font-size: 1.25rem !important;
    padding-left: 0.9rem !important;
    border-left: 3px solid var(--accent) !important;
    margin-top: 1.5rem !important;
    margin-bottom: 0.8rem !important;
}
.stMainBlockContainer h4 {
    color: var(--txt) !important;
    font-weight: 600 !important;
    font-size: 1.05rem !important;
    padding-bottom: 0.45rem !important;
    border-bottom: 2px solid rgba(124,106,255,0.2) !important;
    margin-top: 1.2rem !important;
    margin-bottom: 0.6rem !important;
}
.stMainBlockContainer h5 {
    color: var(--sub) !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    margin-top: 1rem !important;
}

/* ── metric cards (glassmorphism) ── */
.metric-card {
    background: var(--card);
    backdrop-filter: blur(16px) saturate(140%);
    -webkit-backdrop-filter: blur(16px) saturate(140%);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.3rem 1.5rem;
    text-align: center;
    box-shadow: 0 8px 32px rgba(0,0,0,0.05), inset 0 1px 0 rgba(255,255,255,0.8);
    transition: all 0.3s cubic-bezier(.4,0,.2,1);
    min-height: 140px;
    display: flex; flex-direction: column; justify-content: center;
}
.metric-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 40px rgba(0,0,0,0.1), 0 0 0 1px rgba(99,102,241,0.2);
    border-color: rgba(99,102,241,0.3);
}
.metric-label {
    color: var(--sub); font-size: .72rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: .1em;
}
.metric-value {
    font-size: 2.1rem; font-weight: 800;
    margin: .35rem 0; color: var(--txt);
    line-height: 1.1;
}
.metric-sub {
    color: var(--sub); font-size: .76rem; font-weight: 400;
}

/* ── KPI icon indicators ── */
.kpi-icon {
    font-size: 1.3rem; margin-bottom: 0.3rem; display: block;
    filter: drop-shadow(0 0 6px rgba(99,102,241,0.4));
}

/* ── risk colors ── */
.risk-low    { color: var(--green); }
.risk-medium { color: var(--amber); }
.risk-high   { color: var(--red);   }

/* ── insight box (glassmorphism) ── */
.insight-box {
    background: var(--card2);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(99,102,241,0.12);
    border-left: 3px solid var(--accent);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin: .55rem 0;
    font-size: .88rem;
    color: var(--txt);
    transition: all 0.25s ease;
}
.insight-box:hover {
    background: rgba(241,245,249,1.0);
    border-left-color: var(--accent2);
}

/* ── section container ── */
.section-container {
    background: var(--card);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.5rem;
    margin: 1rem 0;
}

/* ── gradient dividers ── */
.gradient-divider {
    height: 1px; border: none; margin: 1.8rem 0;
    background: linear-gradient(90deg, transparent, rgba(99,102,241,0.3), rgba(236,72,153,0.2), transparent);
}

/* ── before/after comparison cards ── */
.comparison-card {
    background: var(--card);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border-radius: var(--radius);
    padding: 1.5rem;
    text-align: center;
    transition: all 0.3s ease;
    min-height: 100px;
}
.comparison-card.before {
    border: 1px solid rgba(239,68,68,0.25);
    box-shadow: 0 4px 24px rgba(239,68,68,0.08);
}
.comparison-card.after {
    border: 1px solid rgba(16,185,129,0.25);
    box-shadow: 0 4px 24px rgba(16,185,129,0.08);
}
.comparison-card:hover { transform: translateY(-2px); }

/* ── progress bars ── */
.stProgress > div > div > div > div {
    border-radius: 8px !important;
    height: 10px !important;
}
.stProgress > div > div > div {
    background: rgba(0,0,0,0.06) !important;
    border-radius: 8px !important;
}

/* ── map containers ── */
iframe {
    border-radius: 14px !important;
    box-shadow: 0 8px 32px rgba(0,0,0,0.15) !important;
    border: 1px solid rgba(99,102,241,0.1) !important;
}

/* ── Streamlit metric override ── */
[data-testid="stMetricValue"] {
    font-weight: 800 !important;
    font-size: 1.8rem !important;
}
[data-testid="stMetricLabel"] {
    font-size: .78rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
    color: var(--sub) !important;
}

/* ── ripple row ── */
.ripple-row {
    display: flex; align-items: center; gap: 1rem;
    padding: .5rem .9rem; border-radius: 10px; margin: .25rem 0;
    background: var(--card);
    backdrop-filter: blur(8px);
    border: 1px solid var(--border);
    transition: background 0.2s;
}
.ripple-row:hover { background: rgba(241,245,249,0.8); }

/* ── urgency banner ── */
.urgency-banner {
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border-radius: 14px;
    padding: 1rem 1.4rem;
    margin-bottom: 1.2rem;
    animation: fadeSlideIn 0.6s ease-out both;
}

/* ── sidebar footer ── */
.sidebar-footer {
    position: fixed; bottom: 0; width: inherit;
    padding: 0.8rem 1.2rem;
    font-size: 0.7rem; color: rgba(139,141,176,0.5);
    border-top: 1px solid rgba(124,106,255,0.08);
    background: linear-gradient(0deg, #0d1020, transparent);
    text-align: center;
}

/* ── branded header ── */
.app-header {
    padding: 0.5rem 0 1rem 0;
    position: relative;
}
.app-header .badge {
    display: inline-block;
    background: rgba(124,106,255,0.12);
    border: 1px solid rgba(124,106,255,0.25);
    border-radius: 20px;
    padding: 0.2rem 0.75rem;
    font-size: 0.7rem;
    font-weight: 600;
    color: var(--accent);
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 0.6rem;
}
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <div class="badge">🔒 Bengaluru Traffic Command</div>
    <h1>🚔 Event-Driven Congestion Forecaster</h1>
    <h3>AI-powered traffic intelligence for planned events &amp; live incidents across Bengaluru</h3>
</div>
<div class="gradient-divider"></div>
""", unsafe_allow_html=True)

# ── Load models (cached) ───────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading AI models…")
def load_models():
    with open(CLOSURE_CLASSIFIER_MODEL_PATH, "rb") as f:
        closure_clf = pickle.load(f)
    with open(SEVERITY_MODEL_PATH, "rb") as f:
        severity_reg = pickle.load(f)
    with open(DURATION_MODEL_PATH, "rb") as f:
        duration_lookup = pickle.load(f)
    with open(ROAD_GRAPH_PATH, "rb") as f:
        G = pickle.load(f)
    with open(FEATURE_NAMES_PATH, "rb") as f:
        feature_names = pickle.load(f)
    return closure_clf, severity_reg, duration_lookup, G, feature_names

closure_clf, severity_reg, duration_lookup, G, feature_names = load_models()

# ── Sidebar – mode selection ───────────────────────────────────────────────────
st.sidebar.markdown("""
<div style="text-align:center;padding:0.8rem 0 0.6rem 0;">
    <div style="font-size:1.6rem;margin-bottom:0.2rem;">🚔</div>
    <div style="font-size:0.85rem;font-weight:800;letter-spacing:0.08em;
                background:linear-gradient(135deg,#7c6aff,#ff6b8a);
                -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
        TRAFFICPULSE
    </div>
    <div style="font-size:0.62rem;color:rgba(139,141,176,0.6);letter-spacing:0.1em;text-transform:uppercase;margin-top:2px;">Command Console</div>
</div>
<div style="height:1px;background:linear-gradient(90deg,transparent,rgba(124,106,255,0.2),transparent);margin:0.4rem 0 1rem 0;"></div>
""", unsafe_allow_html=True)
st.sidebar.markdown("## 🎯 Select Mode")
mode = st.sidebar.radio(
    "What would you like to forecast?",
    ["📅 Planned Event Forecast", "🚨 Live Incident Assessment"],
    label_visibility="collapsed",
)
st.sidebar.markdown('<div style="height:1px;background:linear-gradient(90deg,transparent,rgba(124,106,255,0.15),transparent);margin:0.8rem 0;"></div>', unsafe_allow_html=True)

# ── Shared helpers ─────────────────────────────────────────────────────────────
# FIX: previously this was a hand-picked subset of only 8 corridors, while
# the trained model (models/feature_names.pkl) actually has 21 real
# corridor categories. Using all 21 means the dropdown always matches a
# real, well-represented category the model was trained on.
CORRIDORS = sorted(CORRIDOR_COORDINATES.keys())

# Map corridor display name → duration-lookup key (best effort)
CAUSE_LOOKUP_MAP = {
    "construction":       "construction",
    "public_event":       "public_event",
    "procession":         "procession",
    "vip_movement":       "vip_movement",
    "protest":            "protest",
    "vehicle_breakdown":  "vehicle_breakdown",
    "tree_fall":          "tree_fall",
    "accident":           "accident",
    "pothole":            "pot_holes",
    "waterlogging":       "water_logging",
}

# FIX: CORRIDOR_GEOHASH_MAP (21 corridors, all verified against the actual
# trained graph) is now imported from src.config instead of being a
# hand-rolled 8-entry dict here. The matching NODE_COORDS hack that used to
# live in this spot — geohash2-decoding the synthetic graph IDs — has been
# removed entirely; see the comment above CORRIDOR_COORDINATES in
# src/config.py for the root-cause explanation of why that placed every
# pin near Shivamogga instead of Bengaluru.


def _local_layout(
    center_gh: str, graph: nx.Graph, center_lat: float, center_lon: float,
    max_hops: int = 3, ring_km: float = 1.3,
) -> dict[str, tuple[float, float]]:
    """Position every graph node reachable within `max_hops` of `center_gh`
    on a set of real-world-anchored concentric rings around the actual
    corridor coordinate (center_lat, center_lon).

    The abstract 52-node geohash graph only encodes adjacency (used for the
    spillover/diversion math) — individual node IDs don't correspond to
    real locations. This computes a *display-only* layout, freshly anchored
    at the true corridor location on every call, so the visualisation is
    always centred on the right part of Bengaluru regardless of which
    corridor was selected.
    """
    if center_gh not in graph:
        return {center_gh: (center_lat, center_lon)}

    hop_of = {center_gh: 0}
    frontier = [center_gh]
    hop = 0
    while frontier and hop < max_hops:
        nxt = []
        for n in frontier:
            for nb in graph.neighbors(n):
                if nb not in hop_of:
                    hop_of[nb] = hop + 1
                    nxt.append(nb)
        frontier = nxt
        hop += 1

    by_hop: dict[int, list[str]] = {}
    for node, h in hop_of.items():
        by_hop.setdefault(h, []).append(node)

    coords: dict[str, tuple[float, float]] = {center_gh: (center_lat, center_lon)}
    lat_per_km = 1.0 / 111.0
    lon_per_km = 1.0 / (111.0 * math.cos(math.radians(center_lat)))

    for h, nodes in by_hop.items():
        if h == 0:
            continue
        radius_km = h * ring_km
        n = len(nodes)
        for i, node in enumerate(sorted(nodes)):
            angle = 2 * math.pi * i / n
            d_lat = radius_km * math.sin(angle) * lat_per_km
            d_lon = radius_km * math.cos(angle) * lon_per_km
            coords[node] = (center_lat + d_lat, center_lon + d_lon)

    return coords

def _now_hour() -> int:
    return datetime.now().hour

def _is_peak(h: int) -> int:
    return int((7 <= h <= 10) or (17 <= h <= 20))

def _is_weekend(d: date) -> int:
    return int(d.weekday() >= 5)

def _build_feature_row(raw: dict) -> pd.DataFrame:
    """Build a one-row DataFrame aligned to the saved feature_names."""
    row = {fn: 0.0 for fn in feature_names}
    for k, v in raw.items():
        if k in row:
            row[k] = float(v)
    return pd.DataFrame([row])[feature_names]

def _risk_label(prob: float) -> tuple[str, str]:
    """Return (label, css-class) based on closure probability."""
    if prob < RISK_LOW_THRESHOLD:
        return "LOW", "risk-low"
    elif prob < RISK_HIGH_THRESHOLD:
        return "MEDIUM", "risk-medium"
    else:
        return "HIGH", "risk-high"

def _make_map(event_gh: str, impact_dict: dict, center_lat: float, center_lon: float) -> folium.Map:
    """Build a Folium map showing impact intensity per graph node, anchored
    at the real-world corridor location (center_lat, center_lon)."""
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=13,
        tiles="CartoDB positron",
    )

    node_coords = _local_layout(event_gh, G, center_lat, center_lon)
    max_demand = max(impact_dict.values()) if impact_dict else 1.0

    for node, demand in impact_dict.items():
        if node not in node_coords:
            continue
        lat, lon = node_coords[node]
        ratio = demand / max_demand
        # Colour: green → amber → red
        if ratio < MAP_GREEN_THRESHOLD:
            colour = "#00c896"
        elif ratio < MAP_AMBER_THRESHOLD:
            colour = "#ffb347"
        else:
            colour = "#ff4d6d"

        is_event = node == event_gh
        folium.CircleMarker(
            location=[lat, lon],
            radius=10 if is_event else 7,
            color=colour,
            fill=True,
            fill_color=colour,
            fill_opacity=0.85 if is_event else 0.6,
            weight=3 if is_event else 1,
            popup=folium.Popup(
                f"<b>{node}</b><br>Demand: {demand:.3f}<br>"
                + ("⚠️ EVENT LOCATION" if is_event else ""),
                max_width=180,
            ),
            tooltip=f"{node}: {demand:.2f}",
        ).add_to(m)

    # Highlight event cell (always present — it's the layout's centre)
    folium.Marker(
        location=[center_lat, center_lon],
        icon=folium.Icon(color="red", icon="exclamation-sign", prefix="glyphicon"),
        popup="Event Location",
    ).add_to(m)

    return m


def _make_ripple_map(event_lat: float, event_lon: float, radius_km: float) -> folium.Map:
    """Build a Folium impact-ripple map with concentric severity circles.

    Four rings shrink from the outermost (lightest) to the innermost (darkest),
    matching the spec: [red, orange, yellow, lightgreen] at distance fractions
    [1.0, 0.75, 0.5, 0.25] of radius_km.
    """
    m = folium.Map(
        location=[event_lat, event_lon],
        zoom_start=12,
        tiles="OpenStreetMap",
    )

    ring_fractions = [1.0, 0.75, 0.5, 0.25]
    ring_colours   = ["red", "orange", "yellow", "lightgreen"]
    ring_labels    = ["Outer impact", "High impact", "Severe", "Critical zone"]

    for idx, (frac, colour, label) in enumerate(
        zip(ring_fractions, ring_colours, ring_labels)
    ):
        folium.Circle(
            location=[event_lat, event_lon],
            radius=radius_km * frac * 1000,   # km → metres
            color=colour,
            fill=True,
            fill_color=colour,
            fill_opacity=0.18 + idx * 0.06,   # innermost rings slightly more opaque
            weight=2,
            tooltip=f"{label}: {radius_km * frac:.1f} km radius",
            popup=folium.Popup(
                f"<b>{label}</b><br>Radius: {radius_km * frac:.1f} km",
                max_width=160,
            ),
        ).add_to(m)

    # Event pin on top
    folium.Marker(
        location=[event_lat, event_lon],
        popup=folium.Popup("<b>Event Location</b>", max_width=140),
        tooltip="Event",
        icon=folium.Icon(color="red", icon="exclamation-sign", prefix="glyphicon"),
    ).add_to(m)

    return m


def _radius_from_severity(severity: float) -> float:
    """Convert a 0-1 severity score to a rough congestion radius in km."""
    return MIN_RADIUS_KM + severity * (MAX_RADIUS_KM - MIN_RADIUS_KM)


def _render_forecast_panel(
    fc: dict,
    event_lat: float = BLR_LAT,
    event_lon: float = BLR_LON,
) -> None:
    """Render the shared map + metrics + recommendations panel.

    Parameters
    ----------
    fc       : session_state forecast dict.
    event_lat/lon : coordinates for the event location (corridor-specific).
    """
    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

    col1, col2 = st.columns([3, 1], gap="large")

    radius_km = fc["radius_km"]
    duration  = fc["duration"]

    # ── Column 1: Impact Forecast Map ─────────────────────────────────────────
    with col1:
        st.subheader("📍 Impact Forecast Map")
        ripple_map = _make_ripple_map(event_lat, event_lon, radius_km)
        st_folium(ripple_map, width=1200, height=500, returned_objects=[])

        # Legend
        st.markdown("""
        <div style="display:flex;gap:1.5rem;flex-wrap:wrap;margin-top:.6rem;font-size:.78rem;
                    padding:0.6rem 1rem;background:rgba(22,27,45,0.6);border-radius:10px;
                    border:1px solid rgba(124,106,255,0.1);">
            <span>🔴 Critical zone</span>
            <span>🟠 Severe</span>
            <span>🟡 High impact</span>
            <span>🟢 Outer impact</span>
        </div>""", unsafe_allow_html=True)

    # ── Column 2: Metrics + Deployment Recommendations ────────────────────────
    with col2:
        st.subheader("📊 Summary")
        risk_label, _ = _risk_label(fc["closure_prob"])
        st.metric(
            "Road Closure Probability",
            f"{fc['closure_prob']:.0%}",
            delta=risk_label,
            delta_color=("off" if risk_label == "LOW"
                         else "inverse" if risk_label == "HIGH"
                         else "normal"),
        )
        st.metric("Impact Severity",   f"{fc['severity']:.2f}/1.0")
        st.metric("Congestion Radius", f"{radius_km:.1f} km")
        st.metric("Est. Duration",     f"{duration:.1f} h")

        st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
        st.subheader("🚔 Deployment Recommendations")

        patrol_units     = max(MIN_PATROL_UNITS, int(fc["severity"] * PATROL_MULTIPLIER))
        barricade_points = max(MIN_BARRICADE_POINTS, int(fc["severity"] * BARRICADE_MULTIPLIER))

        st.write(f"**Patrol Units:** {patrol_units}")
        st.write(f"**Barricade Points:** {barricade_points}")
        st.write(f"**Diversion Route:** {DEFAULT_DIVERSION_ROUTE}")

        # Contextual extras
        if fc.get("is_peak"):
            st.write("**Peak Hour:** Deploy at all corridors")
        if fc["closure_prob"] > 0.6:
            st.error("⚠️ High closure risk — pre-position tow vehicles")
        elif fc["closure_prob"] > 0.35:
            st.warning("🟡 Moderate risk — monitoring recommended")
        else:
            st.success("🟢 Low risk — standard patrol sufficient")

    # ── Diversion Simulator ───────────────────────────────────────────────────
    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
    st.subheader("🎯 Diversion Simulator")

    if True:
        # ── Compute real improvement from simulation ──────────────────────
        ev_gh       = CORRIDOR_GEOHASH_MAP.get(fc["corridor"], "tumh")
        # FIX: real-world-anchored layout for this event, replacing the old
        # global NODE_COORDS (which placed everything near Shivamogga — see
        # src/config.py for the root-cause explanation).
        node_coords_local = _local_layout(ev_gh, G, event_lat, event_lon)
        one_hop_s   = set(G.neighbors(ev_gh))
        two_hop_s   = set()
        for _nb in one_hop_s:
            for _nb2 in G.neighbors(_nb):
                if _nb2 != ev_gh and _nb2 not in one_hop_s:
                    two_hop_s.add(_nb2)
        impacted_s    = {ev_gh} | one_hop_s | two_hop_s
        # Pick diversion cells near the impacted area, not randomly across the country
        div_candidates = set()
        for node in impacted_s:
            div_candidates.update(G.neighbors(node))
        
        div_route = list(div_candidates - impacted_s)[:3]
        if len(div_route) < 3:
            # Fallback to geohashes with the same prefix
            div_route = [n for n in G.nodes if n not in impacted_s and n.startswith(ev_gh[:3])][:3]

        # Get real model-computed values
        impact_no_div = simulate_event_impact(ev_gh, fc["severity"], fc["hour"], G)
        impact_div, real_improvement = simulate_with_diversion(
            ev_gh, fc["severity"], fc["hour"], div_route, G
        )
        live_state = get_live_state(fc["hour"], G)

        # Compute actual congestion percentages from simulation
        # "without" = event-cell demand as fraction of max possible
        max_demand = max(impact_no_div.values())
        without_pct = min(0.99, impact_no_div[ev_gh] / max_demand) if max_demand > 0 else 0.5
        with_pct    = min(0.99, impact_div[ev_gh] / max_demand) if max_demand > 0 else 0.3
        delta_pct   = round((without_pct - with_pct) * 100)

        from src.config import DIVERSION_RELIEF_FACTOR, DIVERSION_LOAD_FACTOR
        relief_pct = round((1 - DIVERSION_RELIEF_FACTOR) * 100)
        load_pct   = round((DIVERSION_LOAD_FACTOR - 1) * 100)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Before / After columns ────────────────────────────────────────
        bcol, acol = st.columns(2, gap="large")

        with bcol:
            st.markdown("""
            <div class="comparison-card before">
                <div style="font-size:1.6rem;margin-bottom:0.3rem;">🔴</div>
                <div style="font-weight:800;color:#f87171;font-size:1rem;letter-spacing:0.05em;">WITHOUT DIVERSION</div>
                <div style="font-size:0.72rem;color:var(--sub);margin-top:0.3rem;">Baseline congestion scenario</div>
            </div>""", unsafe_allow_html=True)
            st.markdown('<div style="height:0.5rem;"></div>', unsafe_allow_html=True)
            st.progress(without_pct, text=f"{without_pct:.0%} of peak congestion")

        with acol:
            st.markdown(f"""
            <div class="comparison-card after">
                <div style="font-size:1.6rem;margin-bottom:0.3rem;">🟢</div>
                <div style="font-weight:800;color:#34d399;font-size:1rem;letter-spacing:0.05em;">WITH DIVERSION</div>
                <div style="font-size:0.72rem;color:var(--sub);margin-top:0.3rem;">Via {DEFAULT_DIVERSION_ROUTE}</div>
            </div>""", unsafe_allow_html=True)
            st.markdown('<div style="height:0.5rem;"></div>', unsafe_allow_html=True)
            st.progress(with_pct, text=f"{with_pct:.0%} of peak congestion")

        # ── Improvement summary ───────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        mcol1, mcol2 = st.columns([1, 2])
        with mcol1:
            st.metric(
                label="Improvement",
                value=f"{delta_pct}%",
                delta=f"-{delta_pct}%",
                delta_color="inverse",
                help="Reduction in congestion at event hotspot after diversion is activated",
            )
            st.caption(
                f"Model-computed hotspot relief: **{real_improvement:.1f}%**"
            )
        with mcol2:
            st.success("✅ Diversion recommended for deployment!")
            st.markdown(f"""
            <div class="insight-box" style="margin-top:.5rem">
                <b>Route:</b> {DEFAULT_DIVERSION_ROUTE}<br>
                <small>
                Redirects ~{relief_pct}% of event-cell flow through {len(div_route)} alternate geohash
                {'cell' if len(div_route)==1 else 'cells'} 
                ({', '.join(div_route) if div_route else 'n/a'}).
                Each diversion cell absorbs +{load_pct}% additional load.
                </small>
            </div>""", unsafe_allow_html=True)

        # ── Route comparison map (using real corridor diversion routes) ─────────────
        st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
        st.markdown("##### 🗺️ Route Comparison Map")
        rmap = folium.Map(location=[event_lat, event_lon], zoom_start=12, tiles="OpenStreetMap")

        ev_lat, ev_lon = event_lat, event_lon

        # ── BASELINE ROUTE: straight through the event corridor ──────────────────
        baseline_pts = [
            [ev_lat + 0.03, ev_lon],           # 3 km north
            [ev_lat, ev_lon],                   # Event location
            [ev_lat - 0.03, ev_lon],           # 3 km south
        ]

        folium.PolyLine(
            locations=baseline_pts,
            color="#d85a30",
            weight=6,
            opacity=0.85,
            tooltip="Baseline Route (congested corridor)",
            popup=folium.Popup(
                f"<b>Baseline Route</b><br>Straight through event<br>{without_pct:.0%} congestion",
                max_width=160
            ),
        ).add_to(rmap)

        # ── DIVERSION ROUTE: wide arc computed relative to event location ────
        corridor_name = fc.get("corridor", "Mysore Road")
        diversion_route_name = DEFAULT_DIVERSION_ROUTE  # fallback label
        route_improvement = 0.20

        # Get corridor-specific metadata (bypass direction, route name)
        if corridor_name in DIVERSION_ROUTES:
            route_config = DIVERSION_ROUTES[corridor_name]
            diversion_route_name = route_config["route_name"]
            route_improvement = route_config.get("improvement", 0.20)
            bypass_side = route_config.get("bypass_side", "west")
        else:
            bypass_side = "west"

        # Compute diversion route dynamically based on div_route geohashes
        div_waypoints = [[ev_lat + 0.03, ev_lon]] # Start
        for node in div_route:
            if node in node_coords_local:
                div_waypoints.append(list(node_coords_local[node]))
        div_waypoints.append([ev_lat - 0.03, ev_lon]) # End

        folium.PolyLine(
            locations=div_waypoints,
            color="#3b6d11",
            weight=6,
            opacity=0.85,
            tooltip=f"Diversion Route: {diversion_route_name}",
            popup=folium.Popup(
                f"<b>Diversion Route</b><br>"
                f"{diversion_route_name}<br>"
                f"{with_pct:.0%} congestion<br>"
                f"Improvement: {route_improvement:.0%}",
                max_width=180
            ),
        ).add_to(rmap)

        # Directional markers on the diversion route
        folium.Marker(
            location=div_waypoints[0],
            icon=folium.Icon(color='green', icon='arrow-right', prefix='fa'),
            popup="Start: Diversion Route",
            tooltip="Diversion Start",
        ).add_to(rmap)

        folium.Marker(
            location=div_waypoints[-1],
            icon=folium.Icon(color='green', icon='check', prefix='fa'),
            popup="End: Diversion Route",
            tooltip="Diversion End",
        ).add_to(rmap)

        # ── EVENT MARKER ──────────────────────────────────────────────────────────
        folium.Marker(
            location=[ev_lat, ev_lon],
            icon=folium.Icon(color='red', icon='exclamation', prefix='fa'),
            popup=folium.Popup("<b>Event Location</b>", max_width=120),
            tooltip="Event Location",
        ).add_to(rmap)

        # Legend
        legend_html = f"""
        <div style="position:fixed;bottom:20px;left:20px;z-index:9999;
                    background:rgba(255,255,255,0.9);padding:8px 12px;border-radius:8px;
                    font-size:12px;border:1px solid rgba(99,102,241,0.2);line-height:1.6;
                    color:#0f172a;">
            <span style="color:#d85a30;font-weight:700">&#9472;&#9472;</span> Baseline route ({without_pct:.0%} congestion)<br>
            <span style="color:#3b6d11;font-weight:700">&#9472;&#9472;</span> Diversion route ({with_pct:.0%} congestion)
        </div>"""
        rmap.get_root().html.add_child(folium.Element(legend_html))

        st_folium(rmap, width=None, height=500, returned_objects=[])

# ══════════════════════════════════════════════════════════════════════════════
# MODE A – PLANNED EVENT
# ══════════════════════════════════════════════════════════════════════════════
if mode == "📅 Planned Event Forecast":
    st.sidebar.markdown('<div style="height:1px;background:linear-gradient(90deg,transparent,rgba(124,106,255,0.15),transparent);margin:0.4rem 0 0.8rem 0;"></div>', unsafe_allow_html=True)
    st.sidebar.markdown("### 📅 Plan an Event Response")

    event_cause = st.sidebar.selectbox(
        "Event Type",
        ["construction", "public_event", "procession", "vip_movement", "protest"],
    )
    corridor = st.sidebar.selectbox("Location (Corridor)", CORRIDORS)
    event_date = st.sidebar.date_input("Event Date", value=date.today() + timedelta(days=3))
    event_time_sel = st.sidebar.time_input("Event Time", value=time(10, 0))
    duration_h = st.sidebar.slider("Expected Duration (hours)", 1, 72, 8)

    run_forecast = st.sidebar.button("🔮 FORECAST IMPACT", use_container_width=True)

    if run_forecast:
        today = date.today()
        event_dt = datetime.combine(event_date, event_time_sel)
        days_until = max(0, (event_date - today).days)
        hour = event_time_sel.hour
        is_peak = _is_peak(hour)
        is_weekend = _is_weekend(event_date)
        advance_notice = days_until * 24          # hours ahead

        raw = {
            "is_peak_hour":          is_peak,
            "is_weekend":            is_weekend,
            "advance_notice":        advance_notice,
            "days_until_event":      days_until,
            "minutes_since_reported": 0.0,
            "duration_hours":        float(duration_h),
            f"event_cause_{event_cause}": 1.0,
            f"corridor_{corridor}":  1.0,
        }
        X = _build_feature_row(raw)

        closure_prob  = float(closure_clf.predict_proba(X)[0, 1])
        severity_pred = float(severity_reg.predict(X)[0])
        severity_pred = float(np.clip(severity_pred, 0.0, 1.0))

        lookup_key    = CAUSE_LOOKUP_MAP.get(event_cause, event_cause)
        duration_est  = duration_lookup.get(lookup_key, {}).get("median_hours", duration_h)

        radius_km = _radius_from_severity(severity_pred)

        st.session_state.forecast = {
            "mode":           "planned",
            "closure_prob":   closure_prob,
            "severity":       severity_pred,
            "duration_est":   duration_est,
            "duration":       float(duration_est),     # alias for shared panel
            "radius_km":      radius_km,
            "event_cause":    event_cause,
            "corridor":       corridor,
            "hour":           hour,
            "days_until":     days_until,
            "duration_h":     duration_h,
            "is_peak":        is_peak,
            "is_weekend":     is_weekend,
        }

    # ── Planned mode placeholders ──────────────────────────────────────────────
    if "forecast" not in st.session_state or st.session_state.forecast.get("mode") != "planned":
        col_left, col_right = st.columns([3, 2])
        with col_left:
            st.info("👈 Fill in the event details in the sidebar and click **🔮 FORECAST IMPACT** to see predictions.")
        st.stop()

    fc = st.session_state.forecast

    # ── KPI row ────────────────────────────────────────────────────────────────
    risk_label, risk_cls = _risk_label(fc["closure_prob"])
    cols = st.columns(4)
    kpi_icons = ["🎯", "⚡", "⏱️", "📡"]
    kpis = [
        ("Closure Risk",     f"{fc['closure_prob']*100:.1f}%", risk_label,       risk_cls),
        ("Severity Score",   f"{fc['severity']:.2f}",          "0 = minor · 1 = severe", "metric-sub"),
        ("Est. Duration",    f"{fc['duration_est']:.1f}h",     "from historical median",  "metric-sub"),
        ("Advance Notice",   f"{fc['days_until']}d",           "days before event",       "metric-sub"),
    ]
    for col, (label, value, sub, sub_cls), icon in zip(cols, kpis, kpi_icons):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <span class="kpi-icon">{icon}</span>
                <div class="metric-label">{label}</div>
                <div class="metric-value {risk_cls if label == 'Closure Risk' else ''}">{value}</div>
                <div class="{sub_cls}">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Concentric ripple map + st.metric summary + recommendations ───────────
    # FIX: get real coordinates from CORRIDOR_COORDINATES instead of trying
    # to read from the old global NODE_COORDS (which no longer exists).
    event_gh_planned = CORRIDOR_GEOHASH_MAP.get(fc["corridor"], "tumh")
    ev_lat, ev_lon = CORRIDOR_COORDINATES.get(fc["corridor"], (BLR_LAT, BLR_LON))
    _render_forecast_panel(fc, event_lat=ev_lat, event_lon=ev_lon)

    # ── Geohash ripple detail ──────────────────────────────────────────────────
    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
    st.markdown("#### 🗺️ Geohash Network Ripple Detail")
    gcol_map, gcol_ins = st.columns([3, 2], gap="large")

    event_gh = CORRIDOR_GEOHASH_MAP.get(fc["corridor"], "tumh")
    impact   = simulate_event_impact(event_gh, fc["severity"], fc["hour"], G)

    with gcol_map:
        fmap = _make_map(event_gh, impact, ev_lat, ev_lon)
        st_folium(fmap, width=None, height=380, returned_objects=[])

    with gcol_ins:
        st.markdown("#### 📋 Forecast Summary")
        one_hop = list(G.neighbors(event_gh))
        two_hop_set = set()
        for nb in one_hop:
            for nb2 in G.neighbors(nb):
                if nb2 != event_gh and nb2 not in one_hop:
                    two_hop_set.add(nb2)

        live   = get_live_state(fc["hour"], G)
        direct = impact[event_gh] / live[event_gh] if live.get(event_gh, 0) > 0 else 1.0
        spill1 = sum(impact[n] / live[n] for n in one_hop if live.get(n, 0) > 0) / max(len(one_hop), 1)
        spill2 = sum(impact[n] / live[n] for n in two_hop_set if live.get(n, 0) > 0) / max(len(two_hop_set), 1)

        insights = [
            (f"📍 Event cell <b>{event_gh}</b>",
             f"demand <b>{impact[event_gh]:.3f}</b> ({direct:.2f}x baseline)"),
            (f"🔴 {len(one_hop)} adjacent cells",
             f"avg <b>{spill1:.2f}x</b> baseline (1-hop spillover)"),
            (f"🟡 {len(two_hop_set)} secondary cells",
             f"avg <b>{spill2:.2f}x</b> baseline (2-hop spillover)"),
            ("🕐 Peak hour" if fc["is_peak"] else "🌙 Off-peak",
             "Higher baseline demand" if fc["is_peak"] else "Lower baseline demand"),
            ("📅 Weekend" if fc["is_weekend"] else "📅 Weekday",
             "Leisure traffic pattern" if fc["is_weekend"] else "Commuter traffic pattern"),
        ]
        for title, body in insights:
            st.markdown(
                f'<div class="insight-box"><span>{title}</span><br><small>{body}</small></div>',
                unsafe_allow_html=True,
            )

    # ── Diversion suggestion ───────────────────────────────────────────────────
    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
    st.markdown("#### 🔀 Diversion Scenario Analysis")
    one_hop_set = set(G.neighbors(event_gh))
    two_hop_set2 = set()
    for nb in one_hop_set:
        for nb2 in G.neighbors(nb):
            if nb2 != event_gh and nb2 not in one_hop_set:
                two_hop_set2.add(nb2)
    impacted = {event_gh} | one_hop_set | two_hop_set2
    diversion_candidates = [n for n in sorted(G.nodes) if n not in impacted][:3]

    if diversion_candidates:
        impact_div, improvement_pct = simulate_with_diversion(
            event_gh, fc["severity"], fc["hour"], diversion_candidates, G
        )
        dcol1, dcol2, dcol3 = st.columns(3)
        with dcol1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Diversion Route</div>
                <div class="metric-value" style="font-size:1.1rem">{'  ->  '.join(diversion_candidates)}</div>
                <div class="metric-sub">{len(diversion_candidates)} cells re-routed</div>
            </div>""", unsafe_allow_html=True)
        with dcol2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Hotspot Relief</div>
                <div class="metric-value" style="color:#00c896">{improvement_pct:.1f}%</div>
                <div class="metric-sub">reduction at event cell</div>
            </div>""", unsafe_allow_html=True)
        with dcol3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Route Load Increase</div>
                <div class="metric-value" style="color:#ffb347">+15%</div>
                <div class="metric-sub">per diversion cell</div>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MODE B – LIVE INCIDENT ASSESSMENT
# ══════════════════════════════════════════════════════════════════════════════
else:
    st.sidebar.markdown('<div style="height:1px;background:linear-gradient(90deg,transparent,rgba(124,106,255,0.15),transparent);margin:0.4rem 0 0.8rem 0;"></div>', unsafe_allow_html=True)
    st.sidebar.markdown("### 🚨 Assess an Incident")

    incident_type = st.sidebar.selectbox(
        "Incident Type",
        ["vehicle_breakdown", "tree_fall", "accident", "pothole", "waterlogging", "construction"],
    )
    location = st.sidebar.selectbox("Location (Corridor)", CORRIDORS)
    incident_time_sel = st.sidebar.time_input("Time Reported", value=time(datetime.now().hour, 0))
    reported_mins_ago = st.sidebar.slider("Reported how many minutes ago?", 0, 120, 15)

    run_assess = st.sidebar.button("🚨 ASSESS IMPACT", use_container_width=True)

    if run_assess:
        today     = date.today()
        hour      = incident_time_sel.hour
        is_peak   = _is_peak(hour)
        is_weekend = _is_weekend(today)

        lookup_key   = CAUSE_LOOKUP_MAP.get(incident_type, incident_type)
        duration_est = duration_lookup.get(lookup_key, {}).get("median_hours", 1.0)

        raw = {
            "is_peak_hour":           is_peak,
            "is_weekend":             is_weekend,
            "advance_notice":         0.0,      # unplanned → 0
            "days_until_event":       0.0,
            "minutes_since_reported": float(reported_mins_ago),
            "duration_hours":         float(max(duration_est, 0.5)),
            f"event_cause_{incident_type}": 1.0,
            f"corridor_{location}":   1.0,
        }
        X = _build_feature_row(raw)

        closure_prob  = float(closure_clf.predict_proba(X)[0, 1])
        severity_pred = float(np.clip(severity_reg.predict(X)[0], 0.0, 1.0))

        radius_km = _radius_from_severity(severity_pred)

        st.session_state.forecast = {
            "mode":               "unplanned",
            "closure_prob":       closure_prob,
            "severity":           severity_pred,
            "duration_est":       duration_est,
            "duration":           float(max(duration_est, 0.5)),  # alias for shared panel
            "radius_km":          radius_km,
            "incident_type":      incident_type,
            "corridor":           location,
            "hour":               hour,
            "reported_mins_ago":  reported_mins_ago,
            "is_peak":            is_peak,
            "is_weekend":         is_weekend,
        }

    # ── Unplanned mode placeholders ────────────────────────────────────────────
    if "forecast" not in st.session_state or st.session_state.forecast.get("mode") != "unplanned":
        st.info("👈 Fill in the incident details in the sidebar and click **🚨 ASSESS IMPACT** to see the real-time assessment.")
        st.stop()

    fc = st.session_state.forecast

    # ── Urgency banner ─────────────────────────────────────────────────────────
    risk_label, risk_cls = _risk_label(fc["closure_prob"])
    urgency_colour = {"LOW": "#34d399", "MEDIUM": "#fbbf24", "HIGH": "#f87171"}[risk_label]
    st.markdown(f"""
    <div class="urgency-banner" style="background:linear-gradient(90deg,{urgency_colour}15,{urgency_colour}05,transparent);
                border-left:4px solid {urgency_colour};
                border:1px solid {urgency_colour}25;border-left:4px solid {urgency_colour};">
        <span style="color:{urgency_colour};font-weight:800;font-size:1.05rem;letter-spacing:0.03em;">
        ⚡ {risk_label} URGENCY — {fc['incident_type'].replace('_',' ').upper()} on {fc['corridor']}
        </span><br>
        <span style="color:var(--sub);font-size:.82rem;">
        Reported {fc['reported_mins_ago']} min ago  ·  
        {'Peak hour — heightened impact' if fc['is_peak'] else 'Off-peak — moderate impact'}
        </span>
    </div>""", unsafe_allow_html=True)

    # ── KPI row ────────────────────────────────────────────────────────────────
    cols = st.columns(4)
    kpi_icons_b = ["🎯", "⚡", "⏱️", "🚨"]
    kpis = [
        ("Closure Probability", f"{fc['closure_prob']*100:.1f}%", risk_label, risk_cls),
        ("Severity Score",      f"{fc['severity']:.2f}",          "0–1 scale",                "metric-sub"),
        ("Est. Duration",       f"{fc['duration_est']:.1f}h",     "historical median",         "metric-sub"),
        ("Response Urgency",    f"{fc['reported_mins_ago']} min",  "since first report",        "metric-sub"),
    ]
    for col, (label, value, sub, sub_cls), icon in zip(cols, kpis, kpi_icons_b):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <span class="kpi-icon">{icon}</span>
                <div class="metric-label">{label}</div>
                <div class="metric-value {risk_cls if label == 'Closure Probability' else ''}">{value}</div>
                <div class="{sub_cls}">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Concentric ripple map + st.metric summary + recommendations ───────────
    # FIX: get real coordinates from CORRIDOR_COORDINATES instead of trying
    # to read from the old global NODE_COORDS (which no longer exists).
    event_gh_unplanned = CORRIDOR_GEOHASH_MAP.get(fc["corridor"], "tumh")
    ev_lat, ev_lon = CORRIDOR_COORDINATES.get(fc["corridor"], (BLR_LAT, BLR_LON))
    _render_forecast_panel(fc, event_lat=ev_lat, event_lon=ev_lon)

    # ── Geohash ripple detail ──────────────────────────────────────────────────
    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
    st.markdown("#### 🗺️ Geohash Network Ripple Detail")
    gcol_map, gcol_ins = st.columns([3, 2], gap="large")

    event_gh = CORRIDOR_GEOHASH_MAP.get(fc["corridor"], "tumh")
    impact   = simulate_event_impact(event_gh, fc["severity"], fc["hour"], G)

    with gcol_map:
        st.markdown("**Real-Time Geohash Impact**")
        fmap = _make_map(event_gh, impact, ev_lat, ev_lon)
        st_folium(fmap, width=None, height=380, returned_objects=[])

    with gcol_ins:
        st.markdown("#### 📡 Impact Assessment")

        one_hop = list(G.neighbors(event_gh))
        two_hop_set = set()
        for nb in one_hop:
            for nb2 in G.neighbors(nb):
                if nb2 != event_gh and nb2 not in one_hop:
                    two_hop_set.add(nb2)

        live   = get_live_state(fc["hour"], G)
        direct = impact[event_gh] / live[event_gh] if live.get(event_gh, 0) > 0 else 1.0
        spill1 = sum(impact[n] / live[n] for n in one_hop if live.get(n, 0) > 0) / max(len(one_hop), 1)

        elapsed_h   = fc["reported_mins_ago"] / 60
        remaining_h = max(0.0, fc["duration_est"] - elapsed_h)

        rec_actions = []
        if fc["closure_prob"] > 0.6:
            rec_actions.append("🔴 Deploy emergency response team immediately")
            rec_actions.append("🔀 Activate diversion on adjacent corridors")
        elif fc["closure_prob"] > 0.35:
            rec_actions.append("🟡 Send assessment unit to verify severity")
            rec_actions.append("📢 Issue advisory on dynamic message signs")
        else:
            rec_actions.append("🟢 Monitor via CCTV — likely self-resolving")
        if fc["is_peak"]:
            rec_actions.append("📻 Broadcast alternate routes on radio / app")

        insights = [
            ("Incident severity",
             f"Score <b>{fc['severity']:.2f}</b> -> {direct:.2f}x demand at event cell"),
            (f"{len(one_hop)} neighbouring cells affected",
             f"avg <b>{spill1:.2f}x</b> normal demand"),
            ("Estimated time to clear",
             f"<b>{remaining_h:.1f}h</b> remaining ({fc['duration_est']:.1f}h total)"),
        ]
        for title, body in insights:
            st.markdown(
                f'<div class="insight-box"><span>{title}</span><br><small>{body}</small></div>',
                unsafe_allow_html=True,
            )

        st.markdown("#### 🛡️ Recommended Actions")
        for action in rec_actions:
            st.markdown(
                f'<div class="insight-box" style="border-left-color:#ff6584">{action}</div>',
                unsafe_allow_html=True,
            )

    # ── Diversion panel ────────────────────────────────────────────────────────
    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
    st.markdown("#### 🔀 Rapid Diversion Options")

    one_hop_set = set(G.neighbors(event_gh))
    two_hop_set2 = set()
    for nb in one_hop_set:
        for nb2 in G.neighbors(nb):
            if nb2 != event_gh and nb2 not in one_hop_set:
                two_hop_set2.add(nb2)
    impacted  = {event_gh} | one_hop_set | two_hop_set2
    div_cands = [n for n in sorted(G.nodes) if n not in impacted][:3]

    if div_cands:
        impact_div, improvement_pct = simulate_with_diversion(
            event_gh, fc["severity"], fc["hour"], div_cands, G
        )
        dcol1, dcol2 = st.columns(2)
        with dcol1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Proposed Diversion Cells</div>
                <div class="metric-value" style="font-size:1rem">{'  ->  '.join(div_cands)}</div>
                <div class="metric-sub">auto-selected unimpacted nodes</div>
            </div>""", unsafe_allow_html=True)
        with dcol2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Projected Hotspot Relief</div>
                <div class="metric-value" style="color:#00c896">{improvement_pct:.1f}%</div>
                <div class="metric-sub">reduction at incident cell with diversion active</div>
            </div>""", unsafe_allow_html=True)
