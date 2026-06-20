"""Stage 7 — OSM + HMM road-snapping (DEV_MANUAL §2.5, §2.10).

Snaps the noisy per-frame GPS sequence onto the OSM walking graph with
HMM map-matching (Newson-Krumm Viterbi). Per GPS observation we take
candidate road nodes; the **emission** probability comes from the
GPS-to-candidate distance, the **transition** probability from how
well the on-road distance between consecutive candidates matches
their straight-line distance; Viterbi then picks the most likely
road path.

CLI workflow (DEV_MANUAL §2.10 step 7):

  1. `python -m src.build_walking_graph`   # one-time, produces
                                            # osm_walking.pkl
  2. `python -m src.road_snap`             # consumes
     gps_recovery_full.jsonl, filters to VLM-agreed + top-N POI
     cohort, snaps each video's frame sequence, writes
     road_snapped.jsonl with {video, frame_id, gps_snapped,
     gps_raw, segment_id, segment_bearing, segment_length_m}.

The input filter is **`--top-pois 30`** — only frames whose VLM-
resolved POI (`place_guess`) is among the 30 most common in the
input. That matches the destination pool used by `src/annotate.py`
(§2.7) so HMM works on the same cohort the teacher will annotate.

Pure functions (emission_logp, transition_logp, viterbi) are
unit-tested; `snap()` and the main loop need the osmnx walking
graph + per-frame jsonl input.
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402


def emission_logp(gps_dist_m, sigma_m=20.0):
    """Log emission probability — Gaussian on GPS-to-candidate distance.
    Closer candidate -> higher (less negative) score."""
    return -0.5 * (gps_dist_m / sigma_m) ** 2


def transition_logp(great_circle_m, route_m, beta_m=30.0):
    """Log transition probability — penalises a big gap between the
    on-road distance and the straight-line distance (a detour/teleport)."""
    return -abs(route_m - great_circle_m) / beta_m


def viterbi(obs_states, emit_logp, trans_logp):
    """Generic Viterbi decoder.

    obs_states  : list (per observation) of candidate-state lists
    emit_logp   : f(t, state) -> log emission score
    trans_logp  : f(t, prev_state, state) -> log transition score
    Returns the most-likely state path (one state per observation).
    """
    if not obs_states:
        return []
    V = [{s: emit_logp(0, s) for s in obs_states[0]}]
    back = [{}]
    for t in range(1, len(obs_states)):
        V.append({})
        back.append({})
        for s in obs_states[t]:
            best_prev, best = None, float("-inf")
            for ps in obs_states[t - 1]:
                sc = V[t - 1][ps] + trans_logp(t, ps, s)
                if sc > best:
                    best, best_prev = sc, ps
            V[t][s] = best + emit_logp(t, s)
            back[t][s] = best_prev
    path = [max(V[-1], key=V[-1].get)]
    for t in range(len(obs_states) - 1, 0, -1):
        path.append(back[t][path[-1]])
    return list(reversed(path))


def _haversine_m(la1, lo1, la2, lo2):
    R = 6_371_000.0
    p1, p2 = math.radians(la1), math.radians(la2)
    dphi = math.radians(la2 - la1)
    dlam = math.radians(lo2 - lo1)
    a = (math.sin(dphi / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(dlam / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(a))


def snap(gps_seq, graph_path=None, radius_m=40.0):
    """HMM-snap a [(lat, lon), ...] sequence onto the OSM walking graph.

    Returns a snapped [(lat, lon), ...]. Needs osmnx + a pickled graph.
    """
    import pickle
    import networkx as nx
    import osmnx as ox

    graph_path = graph_path or (config.CITY_DIR / "osm_walking.pkl")
    with open(graph_path, "rb") as f:
        G = pickle.load(f)

    # candidate graph nodes within radius_m of each observation
    from tqdm import tqdm
    obs_states = []
    for lat, lon in tqdm(gps_seq, desc="[road_snap] candidates", unit="obs"):
        node = ox.distance.nearest_nodes(G, lon, lat)
        cands = [node] + [n for n in G.neighbors(node)]
        obs_states.append(cands)

    def emit(t, node):
        olat, olon = gps_seq[t]
        d = _haversine_m(olat, olon, G.nodes[node]["y"], G.nodes[node]["x"])
        return emission_logp(d)

    def trans(t, prev, cur):
        gc = _haversine_m(G.nodes[prev]["y"], G.nodes[prev]["x"],
                          G.nodes[cur]["y"], G.nodes[cur]["x"])
        try:
            route = nx.shortest_path_length(G, prev, cur, weight="length")
        except nx.NetworkXNoPath:
            route = gc * 10           # heavy penalty for unreachable
        return transition_logp(gc, route)

    path = viterbi(obs_states, emit, trans)
    return [(G.nodes[n]["y"], G.nodes[n]["x"]) for n in path]


def _bearing_deg(lat1, lon1, lat2, lon2):
    """Initial compass bearing point 1 -> point 2, in [0, 360)."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    x = math.sin(dl) * math.cos(p2)
    y = (math.cos(p1) * math.sin(p2)
         - math.sin(p1) * math.cos(p2) * math.cos(dl))
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def top_n_pois(rows, n, field="place_guess"):
    """Top-n POI names by frame count in the input rows. Ties broken
    by alphabetic order. Pure — unit-testable."""
    import collections
    counts = collections.Counter((r.get(field) or "") for r in rows)
    counts.pop("", None)
    common = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:n]
    return [name for name, _ in common]


# The 27 canonical "iconic" Zurich POIs from src/poi.py:CANDIDATE_POIS,
# expressed as the OSM-canonical names that actually appear in the
# `place_guess` column of gps_recovery_full.jsonl. Some entries have
# multiple acceptable spellings because OSM has either a longer or a
# nearby-feature variant (e.g. St. Peter the church resolves via the
# adjacent square St. Peterhofstatt; Lake Zurich is "Zürichsee").
FAMOUS_OSM_NAMES = {
    # stations / hills / squares / churches / bridges
    "Zürich Hauptbahnhof", "Hauptbahnhof",
    "Lindenhof", "Paradeplatz", "Münsterhof",
    "Fraumünster", "Grossmünster", "Grossmünsterplatz",
    "St. Peter", "St. Peterhofstatt",
    "Bellevueplatz", "Bellevue", "Sechseläutenplatz", "Bürkliplatz",
    "Quaibrücke", "Münsterbrücke", "Rathausbrücke",
    # civic / culture / stores
    "Rathaus", "Stadthaus", "Opernhaus", "Kunsthaus",
    "Landesmuseum", "Schweizerisches Nationalmuseum",
    "Polyterrasse", "Globus", "Jelmoli",
    # streets / water
    "Bahnhofstrasse", "Niederdorfstrasse", "Limmatquai", "Rennweg",
    "Limmat", "Zürichsee",
}


def famous_pois_with_evidence(rows, min_count=1, field="place_guess"):
    """Famous POIs (from CANDIDATE_POIS) that show up in `rows` at
    least `min_count` times. Returns the OSM names (the form that
    appears in the rows' `field`)."""
    import collections
    counts = collections.Counter((r.get(field) or "") for r in rows)
    return [n for n in sorted(FAMOUS_OSM_NAMES) if counts.get(n, 0) >= min_count]


def _node_latlon(graph, node, to_latlon_xform):
    """Return (lat, lon) for a node, regardless of whether the graph is
    in lat/lon (EPSG:4326) or projected. For projected graphs we use
    the provided pyproj Transformer to convert (x, y) → (lon, lat)."""
    nx_attr = graph.nodes[node]
    if to_latlon_xform is None:
        # graph is already in lat/lon (osmnx default: y=lat, x=lon)
        return nx_attr["y"], nx_attr["x"]
    lon, lat = to_latlon_xform.transform(nx_attr["x"], nx_attr["y"])
    return lat, lon


def _per_video_snap(graph, frames):
    """Snap ONE video's frame sequence (sorted by frame_id) to the
    OSM walking graph. Returns a list of dicts, one per input frame.

    Handles both unprojected (EPSG:4326) and projected (e.g. UTM)
    graphs — projected is preferred so `ox.distance.nearest_nodes`
    can use the fast cKDTree path instead of the scikit-learn
    ball-tree fallback."""
    import networkx as nx
    import osmnx as ox

    if not frames:
        return []
    gps_seq = [tuple(f["gps"]) for f in frames]                # lat, lon

    crs = graph.graph.get("crs")
    is_projected = bool(crs) and "4326" not in str(crs)
    if is_projected:
        from pyproj import Transformer
        to_proj = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        to_latlon = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        # query each (lat, lon) -> (x, y) in graph CRS for nearest_nodes
        xy = [to_proj.transform(lon, lat) for lat, lon in gps_seq]
        nearest = [ox.distance.nearest_nodes(graph, x, y) for x, y in xy]
    else:
        to_latlon = None
        nearest = [ox.distance.nearest_nodes(graph, lon, lat)
                   for lat, lon in gps_seq]

    obs_states = []
    for node in nearest:
        obs_states.append([node] + list(graph.neighbors(node)))

    def _node_xy(n):
        return graph.nodes[n]["x"], graph.nodes[n]["y"]

    def emit(t, node):
        nlat, nlon = _node_latlon(graph, node, to_latlon)
        olat, olon = gps_seq[t]
        return emission_logp(_haversine_m(olat, olon, nlat, nlon))

    def trans(t, prev, cur):
        plat, plon = _node_latlon(graph, prev, to_latlon)
        clat, clon = _node_latlon(graph, cur, to_latlon)
        gc = _haversine_m(plat, plon, clat, clon)
        try:
            route = nx.shortest_path_length(graph, prev, cur,
                                             weight="length")
        except nx.NetworkXNoPath:
            route = gc * 10
        return transition_logp(gc, route)

    path = viterbi(obs_states, emit, trans)

    out = []
    for i, (frame, node) in enumerate(zip(frames, path)):
        nlat, nlon = _node_latlon(graph, node, to_latlon)
        if i + 1 < len(path):
            nxt = path[i + 1]
        elif i > 0:
            nxt = node; node = path[i - 1]
        else:
            nxt = node
        u, v = node, nxt
        if u == v:
            seg_bearing = None
            seg_length = 0.0
            seg_id = None
        else:
            ulat, ulon = _node_latlon(graph, u, to_latlon)
            vlat, vlon = _node_latlon(graph, v, to_latlon)
            seg_bearing = _bearing_deg(ulat, ulon, vlat, vlon)
            seg_length = _haversine_m(ulat, ulon, vlat, vlon)
            seg_id = (int(u), int(v))
        out.append({
            "video": frame["video"],
            "frame_id": frame["frame_id"],
            "gps_snapped": [nlat, nlon],
            "gps_raw": list(frame["gps"]),
            "snap_offset_m": round(_haversine_m(
                frame["gps"][0], frame["gps"][1], nlat, nlon), 2),
            "segment_id": seg_id,
            "segment_bearing": seg_bearing,
            "segment_length_m": round(seg_length, 2),
            "place_guess": frame.get("place_guess", ""),
            "heading": frame.get("heading"),
            "heading_gap": frame.get("heading_gap"),
        })
    return out


def main():
    import argparse
    import collections
    import json
    import pickle
    from tqdm import tqdm

    ap = argparse.ArgumentParser(
        description="HMM road-snap on the VLM-agreed + top-N POI cohort.")
    ap.add_argument("--input",
                    default=str(config.CITY_DIR / "gps_recovery_full.jsonl"),
                    help="per-frame jsonl from src.gps_recovery")
    ap.add_argument("--graph",
                    default=str(config.CITY_DIR / "osm_walking.pkl"),
                    help="pickled osmnx walking graph "
                         "(build with `python -m src.build_walking_graph`)")
    ap.add_argument("--output",
                    default=str(config.CITY_DIR / "road_snapped.jsonl"))
    ap.add_argument("--tier", type=int, choices=[0, 1, 2], default=1,
                    help="only consider rows with this tier "
                         "(default 1 = VLM-agreed only)")
    ap.add_argument("--top-pois", type=int, default=30,
                    help="only snap frames whose place_guess is among "
                         "the top-N most common (default 30; "
                         "matches the §2.7 destination pool). "
                         "0 = no POI filter.")
    ap.add_argument("--include-famous", action="store_true", default=True,
                    help="ALSO include any of the 27 iconic POIs "
                         "(from src/poi.py:CANDIDATE_POIS) that have "
                         "at least --famous-min-count VLM-agreed "
                         "frames. Default ON — restores famous POIs "
                         "(Grossmünster, Fraumünster, etc.) that the "
                         "top-N frame-count filter would otherwise "
                         "drop. Use --no-include-famous to disable.")
    ap.add_argument("--no-include-famous", dest="include_famous",
                    action="store_false")
    ap.add_argument("--famous-min-count", type=int, default=5,
                    help="minimum VLM-agreed frames for a famous POI "
                         "to be added (default 5 — gives meaningful "
                         "per-POI statistics; lower to 1 to include "
                         "every famous POI with any evidence at all).")
    ap.add_argument("--poi-field", default="place_guess",
                    help="which column to use for the top-N filter "
                         "(default place_guess; alt: dino_nearest_name)")
    args = ap.parse_args()

    def _resolve(p):
        """Accept bare filenames (resolve under CITY_DIR) or full paths."""
        path = Path(p)
        if path.exists() or path.is_absolute():
            return path
        in_city = config.CITY_DIR / path.name
        return in_city if in_city.exists() else path

    in_path = _resolve(args.input)
    graph_path = _resolve(args.graph)
    out_path = (Path(args.output) if Path(args.output).is_absolute()
                else config.CITY_DIR / Path(args.output).name)

    if not in_path.exists():
        sys.exit(f"[road_snap] input not found: {in_path}")
    if not graph_path.exists():
        sys.exit(f"[road_snap] OSM walking graph not found: {graph_path}\n"
                 f"  build it first: python -m src.build_walking_graph")

    print(f"[road_snap] input:  {in_path}", flush=True)
    print(f"[road_snap] graph:  {graph_path}", flush=True)
    rows = [json.loads(l) for l in in_path.open(encoding="utf-8")
            if l.strip()]
    if args.tier:
        rows = [r for r in rows if r.get("tier") == args.tier]
    rows = [r for r in rows if r.get("accepted")]
    print(f"[road_snap] after tier={args.tier} + accepted filter: "
          f"{len(rows):,}", flush=True)

    if args.top_pois > 0:
        top = set(top_n_pois(rows, args.top_pois, field=args.poi_field))
        print(f"[road_snap] top-{args.top_pois} {args.poi_field}: "
              f"{sorted(top)[:6]}...", flush=True)

        # Union the iconic 27 famous POIs (src/poi.py:CANDIDATE_POIS)
        # that have at least --famous-min-count VLM-agreed frames.
        # This restores famous landmarks (Grossmünster, Fraumünster,
        # St. Peter, Bellevueplatz, Sechseläutenplatz, Landesmuseum,
        # Grossmünsterplatz, …) that the raw top-N frame-count filter
        # would otherwise drop because they appear less often than
        # busy old-town streets.
        pool = set(top)
        if args.include_famous:
            famous = set(famous_pois_with_evidence(
                rows, min_count=args.famous_min_count,
                field=args.poi_field))
            added = sorted(famous - top)
            pool |= famous
            print(f"[road_snap] +famous (>= {args.famous_min_count} "
                  f"frames, not in top-{args.top_pois}): "
                  f"{len(added)} POIs added — {added}",
                  flush=True)
        rows = [r for r in rows if (r.get(args.poi_field) or "") in pool]
        print(f"[road_snap] after pool filter (top {args.top_pois} + "
              f"famous): {len(rows):,} frames "
              f"({len(pool)} distinct POIs)", flush=True)

    with graph_path.open("rb") as f:
        G = pickle.load(f)
    print(f"[road_snap] graph: {G.number_of_nodes():,} nodes / "
          f"{G.number_of_edges():,} edges", flush=True)

    # group frames per video, sort within each by frame_id so the
    # sequence is chronological
    per_video = collections.defaultdict(list)
    for r in rows:
        per_video[r["video"]].append(r)
    for v in per_video:
        per_video[v].sort(key=lambda r: r["frame_id"])

    n_total = 0
    with out_path.open("w", encoding="utf-8") as fout:
        for video, frames in tqdm(sorted(per_video.items()),
                                   desc="[road_snap]", unit="video"):
            snapped = _per_video_snap(G, frames)
            for row in snapped:
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_total += len(snapped)
            tqdm.write(f"  {video:24s}  {len(snapped):4d} frames "
                       f"snapped")

    print(f"[road_snap] wrote {out_path}  ({n_total:,} rows)", flush=True)
    print(f"  next: python -m src.heading_qc   "
          f"(drop ambiguous-heading / HMM-disagree frames)", flush=True)


if __name__ == "__main__":
    main()
