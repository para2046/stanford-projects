"""Build routes.jsonl — one row per (matched_frame, sampled_destination)
pair, with the OSM-network-routed bearing + GT verb.

Sampling rule per matched frame (1,219 frames in cohort):
  For each of 3 destination slots:
    Roll a band  80% near    (50-500 m)
                 10% medium   (500-1000 m)
                 10% far      (1000-1500 m)
    Sample uniformly among the 21 attractions within that band
    (straight-line distance from frame.gps_snapped to attraction).
    If the band has no candidates → re-roll up to 5 times → fall back
    to any attraction in 50-1500 m.
    If the sampled destination is a duplicate of an earlier slot for
    THIS frame → re-roll the band (up to 5 retries) → if still dup,
    drop the slot.

Routing per (frame, dest):
  src_node  = nearest walking node to frame.gps_snapped
  dst_node  = target.snapped_node_id (point target)
              OR best of multi-targets (shortest path wins)
  path      = nx.shortest_path(G, src_node, dst_node, weight='length')

Verb computation (heading + first-edge bearing):
  first_edge_bearing = bearing(gps(path[0]) → gps(path[1]))
  For verb in {continue, left, right, around}:
    new_heading = (heading + ACTION_DELTA[verb]) mod 360
    error[verb] = |angle_diff(new_heading, first_edge_bearing)|
  GT verb = argmin(error)

Output: data/cities/zurich/a2/routes.jsonl

  python -m src.a2_route
"""

from __future__ import annotations

import collections
import json
import math
import pickle
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config                                       # noqa: E402
from src.a2_attraction_slots import ATTRACTIONS_21  # noqa: E402


SEED = 42
N_PER_FRAME = 3
BANDS = [
    ("near",   50.0,  500.0, 0.80),
    ("medium", 500.0, 1000.0, 0.10),
    ("far",    1000.0, 1500.0, 0.10),
]
ACTION_DELTA = {"continue ahead": 0.0, "turn left": -90.0,
                "turn right": 90.0, "turn around": 180.0}


def _hav(a, b):
    R = 6371000.0
    p1, p2 = math.radians(a[0]), math.radians(b[0])
    dp, dl = p2 - p1, math.radians(b[1] - a[1])
    x = (math.sin(dp / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(x))


def _bearing(a, b):
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    dlon = lon2 - lon1
    x = math.sin(dlon) * math.cos(lat2)
    y = (math.cos(lat1) * math.sin(lat2)
         - math.sin(lat1) * math.cos(lat2) * math.cos(dlon))
    return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0


def _angle_diff(a, b):
    return ((a - b + 540) % 360) - 180


def _make_projector(G):
    """(to_proj, to_latlon, is_projected) — handles the UTM-projected
    walking graph."""
    import pyproj
    crs = G.graph.get("crs")
    is_projected = bool(crs) and "4326" not in str(crs)
    if not is_projected:
        return None, None, False
    return (pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True),
            pyproj.Transformer.from_crs(crs, "EPSG:4326", always_xy=True),
            True)


def _nearest_node(G, lat, lon, to_proj):
    """Use osmnx's spatial index. Handles UTM projection."""
    import osmnx as ox
    if to_proj:
        x, y = to_proj.transform(lon, lat)
        node_id = ox.distance.nearest_nodes(G, x, y)
        nx_, ny_ = G.nodes[node_id]["x"], G.nodes[node_id]["y"]
        offset = math.hypot(x - nx_, y - ny_)
    else:
        node_id = ox.distance.nearest_nodes(G, lon, lat)
        offset = _hav((lat, lon),
                       (G.nodes[node_id]["y"], G.nodes[node_id]["x"]))
    return node_id, offset


def _node_latlon(G, node_id, to_latlon):
    """Get the lat/lon of a node — projecting back if graph is UTM."""
    if to_latlon:
        lon, lat = to_latlon.transform(G.nodes[node_id]["x"],
                                        G.nodes[node_id]["y"])
        return lat, lon
    return G.nodes[node_id]["y"], G.nodes[node_id]["x"]


def _frame_to_dest_distance(frame_gps, target_row):
    """Straight-line distance for band selection. For multi-target
    attractions, use the nearest target."""
    if target_row["target_type"] == "point":
        return _hav(frame_gps, target_row["snapped_gps"])
    else:
        return min(_hav(frame_gps, t["gps"])
                   for t in target_row["multi_targets"])


def _multigraph_edge_length(G, u, v):
    """Min edge length between u and v (handles MultiGraph)."""
    data = G.get_edge_data(u, v)
    if isinstance(data, dict) and "length" in data:
        return float(data["length"])
    return min(float(d.get("length", 0.0)) for d in data.values())


def shortest_path_to_target(G, src_node, target_row, nx):
    """Returns (path, total_length_m) for shortest network path.
    For multi-target, picks the closest target by network distance."""
    if target_row["target_type"] == "point":
        try:
            path = nx.shortest_path(G, src_node,
                                     target_row["snapped_node_id"],
                                     weight="length")
            length = sum(_multigraph_edge_length(G, path[i], path[i+1])
                         for i in range(len(path) - 1))
            return path, length
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None, None
    else:
        best_path, best_len = None, float("inf")
        for t in target_row["multi_targets"]:
            try:
                p = nx.shortest_path(G, src_node, t["node_id"],
                                      weight="length")
                length = sum(_multigraph_edge_length(G, p[i], p[i+1])
                              for i in range(len(p) - 1))
                if length < best_len:
                    best_len = length
                    best_path = p
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
        return (best_path, best_len) if best_path else (None, None)


def gt_verb_from_route(G, path, heading, to_latlon):
    """Pick the verb with the smallest |new_heading - first_edge_bearing|.
    Bearing is computed in lat/lon, so we project node coords back."""
    if not path or len(path) < 2:
        return "continue ahead", 0.0, None, {}
    n0, n1 = path[0], path[1]
    g0 = _node_latlon(G, n0, to_latlon)
    g1 = _node_latlon(G, n1, to_latlon)
    edge_bearing = _bearing(g0, g1)
    errors = {}
    for verb, delta in ACTION_DELTA.items():
        new_h = (heading + delta) % 360
        errors[verb] = round(abs(_angle_diff(new_h, edge_bearing)), 1)
    best_verb = min(errors, key=errors.get)
    return best_verb, edge_bearing, errors[best_verb], errors


def sample_band(rng):
    return rng.choices([b[0] for b in BANDS],
                        weights=[b[3] for b in BANDS])[0]


def band_range(name):
    for b in BANDS:
        if b[0] == name:
            return (b[1], b[2])
    raise ValueError(name)


def main():
    import networkx as nx

    rng = random.Random(SEED)

    # ── load all inputs ────────────────────────────────────────────
    with (config.CITY_DIR / "osm_walking.pkl").open("rb") as f:
        G = pickle.load(f)
    print(f"[route] graph: {G.number_of_nodes():,} nodes / "
          f"{G.number_of_edges():,} edges  CRS={G.graph.get('crs')}")
    to_proj, to_latlon, is_projected = _make_projector(G)
    print(f"[route] projected coords: {is_projected}")

    targets = []
    for line in (config.CITY_DIR / "a2"
                 / "destination_targets.jsonl").open(encoding="utf-8"):
        if not line.strip():
            continue
        targets.append(json.loads(line))
    targets_by_name = {t["attraction"]: t for t in targets}
    print(f"[route] destination targets: {len(targets)}  "
          f"({sum(1 for t in targets if t['target_type']=='point')} point + "
          f"{sum(1 for t in targets if t['target_type']=='multi')} multi)")

    matched = []
    for line in (config.CITY_DIR / "a2"
                 / "GPS_VLM_GEO.jsonl").open(encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("matched"):
            matched.append(r)
    print(f"[route] matched cohort: {len(matched):,}")

    # frame GPS (snapped) + heading
    snap = {}
    for line in (config.CITY_DIR / "a2"
                 / "road_snapped_a2.jsonl").open(encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        snap[(r["video"], r["frame_id"])] = r
    raw_gps = {}
    for line in (config.CITY_DIR
                 / "gps_recovery_full.jsonl").open(encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("g_dino"):
            raw_gps[(r["video"], r["frame_id"])] = r
    heading_v2 = {}
    for line in (config.CITY_DIR / "a2"
                 / "heading_v2.jsonl").open(encoding="utf-8"):
        if not line.strip():
            continue
        d = json.loads(line)
        heading_v2[(d["video"], d["frame_id"])] = d

    # ── pre-compute frame_node for each matched frame ──────────────
    print(f"[route] pre-snapping {len(matched)} frames to graph nodes...")
    frame_node = {}
    for r in matched:
        key = (r["video"], r["frame_id"])
        snap_row = snap.get(key)
        raw_row = raw_gps.get(key)
        gps = (tuple(snap_row["gps_snapped"]) if snap_row
               else (tuple(raw_row["g_dino"]) if raw_row else None))
        if gps is None:
            continue
        node_id, off = _nearest_node(G, gps[0], gps[1], to_proj)
        frame_node[key] = {
            "node": int(node_id),
            "snap_offset_m": round(off, 1),
            "gps": gps,
            "raw_gps": (tuple(raw_row["g_dino"]) if raw_row else None),
        }

    # ── per-frame, sample 3 destinations + compute route + verb ────
    out_path = config.CITY_DIR / "a2" / "routes.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_pairs = 0
    n_no_path = 0
    band_counter = collections.Counter()
    verb_counter = collections.Counter()
    dist_hist = collections.Counter()
    per_attr_count = collections.Counter()

    with out_path.open("w", encoding="utf-8") as fout:
        for r in matched:
            key = (r["video"], r["frame_id"])
            fn = frame_node.get(key)
            if not fn:
                continue
            hv2 = heading_v2.get(key, {})
            heading = hv2.get("heading_v2")
            if heading is None:
                heading = (raw_gps.get(key) or {}).get("heading") or 0.0

            already_sampled = set()
            for slot in range(N_PER_FRAME):
                # sample band + destination, re-roll on dup
                dest_name = None
                band_chosen = None
                for retry in range(6):
                    band = sample_band(rng)
                    lo, hi = band_range(band)
                    candidates = []
                    for t in targets:
                        d = _frame_to_dest_distance(fn["gps"], t)
                        if lo <= d < hi and t["attraction"] not in already_sampled:
                            candidates.append((t["attraction"], d))
                    if candidates:
                        dest_name, dist = rng.choice(candidates)
                        band_chosen = band
                        break
                if dest_name is None:
                    # final fallback: any attraction in 50-1500m,
                    # not duplicated
                    fb = [(t["attraction"],
                            _frame_to_dest_distance(fn["gps"], t))
                          for t in targets
                          if t["attraction"] not in already_sampled]
                    fb = [(n, d) for n, d in fb if 50 <= d < 1500]
                    if not fb:
                        break
                    dest_name, dist = rng.choice(fb)
                    band_chosen = "fallback"

                already_sampled.add(dest_name)
                band_counter[band_chosen] += 1

                target = targets_by_name[dest_name]

                # route
                path, total_len = shortest_path_to_target(
                    G, fn["node"], target, nx)
                if path is None:
                    n_no_path += 1
                    continue

                gt_verb, edge_bearing, verb_error, all_errors = \
                    gt_verb_from_route(G, path, heading, to_latlon)

                # great-circle for comparison
                if target["target_type"] == "point":
                    dest_gps_for_gc = target["snapped_gps"]
                else:
                    # use the closest multi-target
                    nearest_t = min(
                        target["multi_targets"],
                        key=lambda t: _hav(fn["gps"], t["gps"]))
                    dest_gps_for_gc = nearest_t["gps"]
                gc_bearing = _bearing(fn["gps"], dest_gps_for_gc)
                bearing_diff = abs(_angle_diff(edge_bearing or gc_bearing,
                                                gc_bearing))

                # decimal-band the straight-line distance for histogram
                dist_band = (int(dist // 100) * 100,
                              int(dist // 100) * 100 + 100)
                dist_hist[dist_band] += 1
                verb_counter[gt_verb] += 1
                per_attr_count[dest_name] += 1

                row = {
                    "video": r["video"], "frame_id": r["frame_id"],
                    "current_gps_raw":     (list(fn["raw_gps"])
                                             if fn["raw_gps"] else None),
                    "current_gps_snapped": list(fn["gps"]),
                    "current_node_id":     fn["node"],
                    "current_snap_offset_m": fn["snap_offset_m"],
                    "heading":             round(heading, 1),
                    "destination":         dest_name,
                    "destination_zh":      target["zh"],
                    "destination_kind":    target["kind"],
                    "destination_target_type": target["target_type"],
                    "target_gps":          dest_gps_for_gc,
                    "target_node_id":      path[-1],
                    "route_node_ids":      [int(n) for n in path],
                    "n_segments":          len(path) - 1,
                    "route_bearing_network": round(edge_bearing, 1)
                                              if edge_bearing else None,
                    "route_bearing_great_circle": round(gc_bearing, 1),
                    "bearing_diff_deg":    round(bearing_diff, 1),
                    "route_distance_m":    round(total_len, 1),
                    "frame_dest_distance_m_raw": round(dist, 1),
                    "sampling_band":       band_chosen,
                    "first_segment_length_m": (
                        round(_multigraph_edge_length(G, path[0], path[1]), 1)
                        if len(path) >= 2 else 0.0),
                    "gt_verb":             gt_verb,
                    "verb_error_deg":      verb_error,
                    "verb_errors":         all_errors,
                }
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_pairs += 1

    print(f"[route] wrote {out_path}  ({n_pairs:,} (frame, dest) pairs)")
    print(f"[route] no-path failures:  {n_no_path}")

    # ── summary ────────────────────────────────────────────────────
    print()
    print("=" * 80)
    print("BAND DISTRIBUTION (target was 80/10/10)")
    print("=" * 80)
    total = sum(band_counter.values())
    for b in ["near", "medium", "far", "fallback"]:
        c = band_counter[b]
        print(f"  {b:<10s}  {c:>5d}  ({100*c/max(1,total):.1f} %)")

    print()
    print("=" * 80)
    print("GT VERB DISTRIBUTION")
    print("=" * 80)
    for v, c in verb_counter.most_common():
        print(f"  {v:<16s}  {c:>5d}  ({100*c/max(1,n_pairs):.1f} %)")

    print()
    print("=" * 80)
    print("PER-ATTRACTION PAIR COUNT")
    print("=" * 80)
    for en, _zh, *_ in ATTRACTIONS_21:
        c = per_attr_count[en]
        print(f"  {en:<22s}  {c:>5d}")


if __name__ == "__main__":
    main()
