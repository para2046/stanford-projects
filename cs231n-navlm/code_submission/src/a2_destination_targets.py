"""Build destination_targets.jsonl — the canonical routing target per
attraction.

HYBRID strategy:
  POINT-TARGET (16 attractions, kind not in {water, street}):
      One target = canonical GPS snapped to nearest walking-graph node.
      Used for buildings, squares, bridges, museums, etc.

  MULTI-TARGET (5 long features, kind in {water, street}):
      Lake Zurich · Limmat river · Bahnhofstrasse · Niederdorfstrasse · Limmatquai
      Targets = list of matched-cohort panos tagged with this attraction.
      "Arrived" = reach ANY of them.

Output: data/cities/zurich/a2/destination_targets.jsonl  (21 rows)

  python -m src.a2_destination_targets
"""

from __future__ import annotations

import collections
import json
import math
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config                                       # noqa: E402
from src.a2_attraction_slots import ATTRACTIONS_21  # noqa: E402


CANON = {en for en, *_ in ATTRACTIONS_21}
MULTI_KINDS = {"water", "street"}


def _hav(a, b):
    R = 6371000.0
    p1, p2 = math.radians(a[0]), math.radians(b[0])
    dp, dl = p2 - p1, math.radians(b[1] - a[1])
    x = (math.sin(dp / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(x))


def _make_projector(G):
    """Returns (to_proj, to_latlon, is_projected). The graph stored on
    disk is UTM-projected; we need to project lat/lon queries to that
    CRS before nearest-node lookup."""
    import pyproj
    crs = G.graph.get("crs")
    is_projected = bool(crs) and "4326" not in str(crs)
    if not is_projected:
        return None, None, False
    to_proj = pyproj.Transformer.from_crs("EPSG:4326", crs,
                                            always_xy=True)
    to_latlon = pyproj.Transformer.from_crs(crs, "EPSG:4326",
                                              always_xy=True)
    return to_proj, to_latlon, True


def nearest_node(G, lat, lon, to_proj):
    """Use osmnx's spatial index. Handles UTM projection."""
    import osmnx as ox
    if to_proj:
        x, y = to_proj.transform(lon, lat)        # always_xy → (lon, lat)
        node_id = ox.distance.nearest_nodes(G, x, y)
        # offset (m) — graph is in UTM so subtract UTM coords
        nx_, ny_ = G.nodes[node_id]["x"], G.nodes[node_id]["y"]
        offset = math.hypot(x - nx_, y - ny_)
    else:
        node_id = ox.distance.nearest_nodes(G, lon, lat)
        offset = _hav((lat, lon),
                       (G.nodes[node_id]["y"], G.nodes[node_id]["x"]))
    return node_id, offset


def node_latlon(G, node_id, to_latlon):
    """Get the lat/lon of a node — projecting back if graph is UTM."""
    if to_latlon:
        lon, lat = to_latlon.transform(G.nodes[node_id]["x"],
                                        G.nodes[node_id]["y"])
        return lat, lon
    return G.nodes[node_id]["y"], G.nodes[node_id]["x"]


def frame_attractions(r):
    out = set()
    for a in r["list_a_gps"].get("attractions", []):
        if a in CANON: out.add(a)
    for a in r["list_b_vlm"].get("attractions", []):
        if a in CANON: out.add(a)
    for m in r["matches"]:
        for nm in [m["gps_name"], m["vlm_name"]]:
            if nm in CANON: out.add(nm)
    return out


def main():
    # ── load OSM walking graph ─────────────────────────────────────
    with (config.CITY_DIR / "osm_walking.pkl").open("rb") as f:
        G = pickle.load(f)
    print(f"[dest_targets] graph: {G.number_of_nodes():,} nodes / "
          f"{G.number_of_edges():,} edges  CRS={G.graph.get('crs')}")
    to_proj, to_latlon, is_projected = _make_projector(G)
    print(f"[dest_targets] projected coords: {is_projected}")

    # ── load matched cohort + frame GPS (for the multi-targets) ────
    matched = []
    for line in (config.CITY_DIR / "a2"
                 / "GPS_VLM_GEO.jsonl").open(encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("matched"):
            matched.append(r)

    # frame GPS (prefer HMM-snapped, fall back to raw)
    snap = {}
    for line in (config.CITY_DIR / "a2"
                 / "road_snapped_a2.jsonl").open(encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        snap[(r["video"], r["frame_id"])] = tuple(r["gps_snapped"])
    raw_gps = {}
    for line in (config.CITY_DIR
                 / "gps_recovery_full.jsonl").open(encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("g_dino"):
            raw_gps[(r["video"], r["frame_id"])] = tuple(r["g_dino"])

    def frame_gps(key):
        return snap.get(key) or raw_gps.get(key)

    # ── collect panos per multi-target attraction ───────────────────
    # group by pano_id so we get one target per unique pano
    panos_per_attr = collections.defaultdict(dict)
    for r in matched:
        key = (r["video"], r["frame_id"])
        g = frame_gps(key)
        if not g:
            continue
        for en in frame_attractions(r):
            # use one representative frame per pano
            pano = "_".join(key)        # we don't have pano_id directly here
            panos_per_attr[en].setdefault(pano, {
                "video": key[0], "frame_id": key[1], "gps": list(g),
            })

    # ── build the rows ─────────────────────────────────────────────
    out_path = config.CITY_DIR / "a2" / "destination_targets.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    point_count = 0
    multi_count = 0
    multi_target_counts = []

    with out_path.open("w", encoding="utf-8") as fout:
        for en, zh, lat, lon, kind in ATTRACTIONS_21:
            row = {
                "attraction": en, "zh": zh, "kind": kind,
                "canonical_gps": [lat, lon],
            }
            if kind in MULTI_KINDS:
                # MULTI-TARGET: matched-cohort frames tagged with this attr
                panos = list(panos_per_attr.get(en, {}).values())
                targets = []
                for p in panos:
                    node_id, off = nearest_node(G, p["gps"][0], p["gps"][1], to_proj)
                    targets.append({
                        "video": p["video"], "frame_id": p["frame_id"],
                        "node_id": int(node_id),
                        "gps": p["gps"],
                        "snap_offset_m": round(off, 1),
                    })
                # also include the canonical GPS as a target (in case
                # cohort missed an obvious spot)
                canon_node, canon_off = nearest_node(G, lat, lon, to_proj)
                if not any(t["node_id"] == int(canon_node) for t in targets):
                    targets.append({
                        "video": "<canonical>",
                        "frame_id": "<canonical>",
                        "node_id": int(canon_node),
                        "gps": [lat, lon],
                        "snap_offset_m": round(canon_off, 1),
                    })
                row.update({
                    "target_type": "multi",
                    "n_targets": len(targets),
                    "multi_targets": targets,
                    "is_routable": len(targets) > 0,
                })
                multi_count += 1
                multi_target_counts.append((en, len(targets)))
            else:
                # POINT-TARGET: snap canonical to nearest walking node
                node_id, off = nearest_node(G, lat, lon, to_proj)
                node_lat, node_lon = node_latlon(G, node_id, to_latlon)
                row.update({
                    "target_type": "point",
                    "snapped_node_id": int(node_id),
                    "snapped_gps": [node_lat, node_lon],
                    "snap_offset_m": round(off, 1),
                    "is_routable": True,
                })
                point_count += 1
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[dest_targets] wrote {out_path}")
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"point-target attractions: {point_count}")
    print(f"multi-target attractions: {multi_count}")
    print()
    print("--- multi-target counts ---")
    for en, n in multi_target_counts:
        print(f"  {en:<22s}  {n} targets")

    print()
    print("--- per-attraction summary (line-by-line) ---")
    with out_path.open(encoding="utf-8") as fin:
        for line in fin:
            r = json.loads(line)
            if r["target_type"] == "point":
                print(f"  {r['attraction']:<22s} POINT  "
                      f"snap_offset = {r['snap_offset_m']:>5.1f} m")
            else:
                print(f"  {r['attraction']:<22s} MULTI  "
                      f"n_targets = {r['n_targets']}")


if __name__ == "__main__":
    main()
