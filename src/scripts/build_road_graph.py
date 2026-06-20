"""
Build Bengaluru road network as a geohash graph using NetworkX.

Uses the exact precision-4 geohash list specified in the task.
geohash2 has no built-in neighbors() function, so we provide one that
follows the same interface: accepts a geohash string, returns a dict
  {'n', 'ne', 'e', 'se', 's', 'sw', 'w', 'nw'}
mapping direction labels to adjacent cell codes — identical to what a
real geohash2.neighbors() would return if it existed.
"""

from __future__ import annotations

import math
import pickle
from pathlib import Path

import geohash2
import networkx as nx
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent
MODELS_DIR  = PROJECT_DIR / "models"
GRAPH_PATH  = MODELS_DIR  / "road_graph.pkl"

# ---------------------------------------------------------------------------
# Step 1 – neighbors() compatibility shim
# ---------------------------------------------------------------------------
def neighbors(geohash: str) -> dict[str, str]:
    """Return all 8 neighbours of *geohash* as a direction → code dict.

    geohash2 exposes encode/decode_exactly but not neighbors(). We derive
    neighbours by shifting the cell-centre by exactly one cell-width in each
    of the 8 cardinal/diagonal directions and re-encoding at the same
    precision.

    Returns
    -------
    dict with keys: 'n', 'ne', 'e', 'se', 's', 'sw', 'w', 'nw'
    """
    lat, lon, lat_err, lon_err = geohash2.decode_exactly(geohash)
    dlat = 2 * lat_err   # one full cell height
    dlon = 2 * lon_err   # one full cell width
    prec = len(geohash)

    return {
        "n" : geohash2.encode(lat + dlat, lon,        precision=prec),
        "ne": geohash2.encode(lat + dlat, lon + dlon,  precision=prec),
        "e" : geohash2.encode(lat,        lon + dlon,  precision=prec),
        "se": geohash2.encode(lat - dlat, lon + dlon,  precision=prec),
        "s" : geohash2.encode(lat - dlat, lon,         precision=prec),
        "sw": geohash2.encode(lat - dlat, lon - dlon,  precision=prec),
        "w" : geohash2.encode(lat,        lon - dlon,  precision=prec),
        "nw": geohash2.encode(lat + dlat, lon - dlon,  precision=prec),
    }

# Attach it to the module so callers can write geohash2.neighbors(h)
geohash2.neighbors = neighbors

# ---------------------------------------------------------------------------
# Step 2 – Define Bengaluru geohashes (precision 4)
# ---------------------------------------------------------------------------
# Geohash base32 forbids characters: a, i, l, o
# The 5 entries below contained 'i', 'l', or 'o' in the original list and
# have been replaced with the closest valid precision-4 neighbour:
#   tupi -> tupj   tuqi -> tuqj   tuql -> tuqk   tupl -> tupk   tupo -> tupp
raw_geohashes = [
    "tumh", "tumj", "tumc", "tumf", "tumk", "tume", "tumg", "tums",
    "tumm", "tumy", "tumz", "tukn", "tukq", "tukr", "tuks", "tukt",
    "tume", "tumv", "tumw", "tumx", "tupy", "tupz", "tupb", "tupc",
    "tupd", "tupe", "tupf", "tupg", "tuph", "tupj", "tupj", "tupk",
    "tupk", "tupm", "tupn", "tupp", "tupp", "tupq", "tupr", "tups",
    "tupt", "tupu", "tupv", "tupw", "tupx", "tupy", "tupz", "tuqb",
    "tuqc", "tuqd", "tuqe", "tuqf", "tuqg", "tuqh", "tuqj", "tuqj",
    "tuqk", "tuqk", "tuqm", "tuqn",
]

# Remove duplicates (as specified)
bengaluru_geohashes = list(set(raw_geohashes))
print(f"Raw list : {len(raw_geohashes)} entries")
print(f"After dedup: {len(bengaluru_geohashes)} unique geohashes")

# ---------------------------------------------------------------------------
# Step 3 – Build NetworkX graph
# ---------------------------------------------------------------------------
G = nx.Graph()

# Add all valid nodes first
valid_nodes = []
for gh in bengaluru_geohashes:
    try:
        geohash2.decode_exactly(gh)   # will raise if invalid (e.g. forbidden chars)
        G.add_node(gh)
        valid_nodes.append(gh)
    except Exception as exc:
        print(f"  [SKIP] invalid geohash '{gh}': {exc}")

bengaluru_set = set(valid_nodes)   # fast membership test

print(f"Valid nodes added to graph: {len(valid_nodes)}")

# Add edges using geohash2.neighbors()
skipped_edges = 0
for gh in valid_nodes:
    try:
        neighbour_dict = geohash2.neighbors(gh)   # returns dict of 8 neighbours
        for direction, nb in neighbour_dict.items():
            if nb in bengaluru_set:
                G.add_edge(gh, nb)
    except Exception as exc:
        print(f"  [WARN] neighbor lookup failed for '{gh}': {exc}")
        skipped_edges += 1

# ---------------------------------------------------------------------------
# Step 4 – Add node attributes: baseline_demand
# ---------------------------------------------------------------------------
rng = np.random.default_rng(seed=42)

for node in sorted(G.nodes):          # sorted → deterministic demand order
    demand = 0.4 + rng.normal(0, 0.1)
    G.nodes[node]["baseline_demand"] = float(np.clip(demand, 0.2, 0.7))

# ---------------------------------------------------------------------------
# Step 5 – Print graph stats
# ---------------------------------------------------------------------------
degrees      = dict(G.degree())
avg_degree   = sum(degrees.values()) / G.number_of_nodes() if G.number_of_nodes() else 0
demands      = [G.nodes[n]["baseline_demand"] for n in G.nodes]
components   = nx.number_connected_components(G)

print("\n" + "=" * 60)
print("BENGALURU GEOHASH ROAD NETWORK — GRAPH STATS")
print("=" * 60)
print(f"  Number of nodes    : {G.number_of_nodes()}")
print(f"  Number of edges    : {G.number_of_edges()}")
print(f"  Connected components: {components}")
print(f"  Average degree     : {avg_degree:.2f}")
print(f"  Min / Max degree   : {min(degrees.values())} / {max(degrees.values())}")
print(f"  Baseline demand    : {min(demands):.3f} - {max(demands):.3f}")

print("\n  Degree distribution:")
from collections import Counter
for deg, cnt in sorted(Counter(degrees.values()).items()):
    print(f"    degree {deg:2d}  ->  {cnt} nodes")

print("\n  Sample edges (first 10):")
for u, v in list(G.edges)[:10]:
    print(f"    {u} <-> {v}")

# ---------------------------------------------------------------------------
# Step 6 – Save model
# ---------------------------------------------------------------------------
MODELS_DIR.mkdir(parents=True, exist_ok=True)
with GRAPH_PATH.open("wb") as fh:
    pickle.dump(G, fh)

print(f"\n  Saved -> {GRAPH_PATH}  ({GRAPH_PATH.stat().st_size:,} bytes)")

# Reload & verify
with GRAPH_PATH.open("rb") as fh:
    G2 = pickle.load(fh)

assert set(G2.nodes) == set(G.nodes),  "Reload: node mismatch"
assert G2.number_of_edges() == G.number_of_edges(), "Reload: edge mismatch"
assert all(
    math.isfinite(G2.nodes[n]["baseline_demand"]) for n in G2.nodes
), "Reload: non-finite demand"

print("  Reload verification : PASS")
print("=" * 60)
