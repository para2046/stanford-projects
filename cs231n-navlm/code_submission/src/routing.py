"""Routing — OSM walking route + relative actions (DEV_MANUAL §2.6).

`plan_route` finds the OSM walking route between two GPS points; the
pure geometry helpers turn absolute bearings into the relative action
verb an instruction must use.

Pure functions (bearing_deg, angle_diff, action_for, closed_loop_delta,
distance_phrase, ACTION_DELTA) are unit-tested; `plan_route` needs the
osmnx walking graph.
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

# relative verb -> heading change it implies (used by the closed-loop check)
ACTION_DELTA = {
    "continue ahead": 0.0,
    "turn left": -90.0,
    "turn right": 90.0,
    "turn around": 180.0,
}


def bearing_deg(lat1, lon1, lat2, lon2) -> float:
    """Initial compass bearing point 1 -> point 2, degrees in [0, 360)."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    x = math.sin(dl) * math.cos(p2)
    y = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def angle_diff(a, b) -> float:
    """Signed smallest difference a - b, in (-180, 180]."""
    return ((a - b + 540) % 360) - 180


def action_for(turn_deg) -> str:
    """Relative action verb for a signed bearing change (DEV_MANUAL §2.6).
    |Δ|≤35 ahead · |Δ|>135 around · else left/right by sign."""
    a = turn_deg
    if abs(a) <= 35:
        return "continue ahead"
    if abs(a) > 135:
        return "turn around"
    return "turn left" if a < 0 else "turn right"


def closed_loop_delta(heading, action, route_bearing) -> float:
    """The verifier's δ: |heading + ACTION_DELTA[action] − route_bearing|."""
    return abs(angle_diff(heading + ACTION_DELTA[action], route_bearing))


def distance_phrase(metres) -> str:
    """TTS-friendly distance phrase — keeps raw numbers out of answers."""
    if metres < 30:
        return "just a few steps"
    if metres < 100:
        return "about a block"
    if metres < 250:
        return "a couple of blocks"
    if metres < 500:
        return "a few blocks"
    return "several blocks"


def plan_route(start_gps, dest_gps, graph_path=None):
    """OSM walking route between two (lat, lon) points.

    Returns {distance_m, first_seg_bearing, n_nodes, route_latlon[]} or
    None if unroutable. Needs osmnx + a pickled walking graph.
    """
    import pickle
    import networkx as nx
    import osmnx as ox

    graph_path = graph_path or (config.CITY_DIR / "osm_walking.pkl")
    with open(graph_path, "rb") as f:
        G = pickle.load(f)

    sn = ox.distance.nearest_nodes(G, start_gps[1], start_gps[0])
    dn = ox.distance.nearest_nodes(G, dest_gps[1], dest_gps[0])
    try:
        nodes = nx.shortest_path(G, sn, dn, weight="length")
    except nx.NetworkXNoPath:
        return None

    pts = [(G.nodes[n]["y"], G.nodes[n]["x"]) for n in nodes]
    dist = 0.0
    for a, b in zip(pts, pts[1:]):
        dist += _haversine_m(a[0], a[1], b[0], b[1])
    first_bearing = (bearing_deg(pts[0][0], pts[0][1], pts[1][0], pts[1][1])
                     if len(pts) > 1 else 0.0)
    return {"distance_m": dist, "first_seg_bearing": first_bearing,
            "n_nodes": len(pts), "route_latlon": pts}


def _haversine_m(la1, lo1, la2, lo2) -> float:
    R = 6_371_000.0
    p1, p2 = math.radians(la1), math.radians(la2)
    dphi = math.radians(la2 - la1)
    dlam = math.radians(lo2 - lo1)
    a = (math.sin(dphi / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(dlam / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(a))
