"""Build the OSM walking-graph pickle that road_snap.py consumes.

Runs once. Downloads central-Zurich's pedestrian network from
OpenStreetMap (via osmnx Overpass), bakes it to a NetworkX MultiDiGraph,
and pickles it to `data/cities/zurich/osm_walking.pkl`.

  python -m src.build_walking_graph                       # default bbox
  python -m src.build_walking_graph --margin-m 600        # wider buffer
  python -m src.build_walking_graph --force               # overwrite

The bbox is `config.POI_BBOX` + `--margin-m` metres on each side so
that routes leaving the POI box (e.g. continuing past Hauptbahnhof or
along the lake-front) still have walkable geometry. Default margin
matches `config.SV_MARGIN_M` (300 m) for consistency with the SV
crawl box.
"""

import argparse
import math
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config                                  # noqa: E402


def _expand_bbox(bbox, margin_m):
    """(W, S, E, N) + margin in metres, using ~111 km / deg latitude
    and cos(mean_lat)-corrected longitude. Returns (W, S, E, N)."""
    W, S, E, N = bbox
    dlat = margin_m / 111_000.0
    mean_lat = (S + N) / 2.0
    dlon = margin_m / (111_000.0 * max(0.1, math.cos(math.radians(mean_lat))))
    return (W - dlon, S - dlat, E + dlon, N + dlat)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--margin-m", type=float, default=config.SV_MARGIN_M,
                    help="metres of buffer around config.POI_BBOX "
                         f"(default {config.SV_MARGIN_M})")
    ap.add_argument("--output", default=str(config.CITY_DIR / "osm_walking.pkl"))
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing pickle")
    args = ap.parse_args()

    out_path = Path(args.output)
    if out_path.exists() and not args.force:
        sys.exit(f"{out_path} already exists — use --force to overwrite")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    bbox = _expand_bbox(config.POI_BBOX, args.margin_m)
    W, S, E, N = bbox
    print(f"[build_walking_graph] bbox (W,S,E,N) = "
          f"({W:.4f}, {S:.4f}, {E:.4f}, {N:.4f})  "
          f"[POI_BBOX + {args.margin_m:.0f} m]", flush=True)

    import osmnx as ox
    # osmnx 2.x uses bbox=(W,S,E,N); 1.x uses (N,S,E,W) positional args.
    try:
        G = ox.graph_from_bbox(bbox=(W, S, E, N), network_type="walk")
    except TypeError:
        G = ox.graph_from_bbox(N, S, E, W, network_type="walk")
    print(f"[build_walking_graph] graph: "
          f"{G.number_of_nodes():,} nodes · "
          f"{G.number_of_edges():,} edges", flush=True)

    # Project to UTM so `ox.distance.nearest_nodes` can use a fast
    # cKDTree without the scikit-learn ball-tree fallback (which is
    # only invoked on unprojected lat/lon graphs). We keep BOTH the
    # original lat/lon coords (on every node, as `y` and `x`) AND a
    # _projected_ copy stored under attribute "_projected" so callers
    # can pick — road_snap.py uses the projected one for snap, then
    # reads the lat/lon back for output.
    G_proj = ox.project_graph(G)
    print(f"[build_walking_graph] projected to UTM: "
          f"{G_proj.graph.get('crs')}", flush=True)
    # We pickle the PROJECTED graph; lat/lon are re-derivable by
    # projecting back to EPSG:4326 if ever needed. For our snap loop
    # only the relative geometry matters, and we then convert each
    # chosen node's UTM (x, y) back to lat/lon for the output file.
    with out_path.open("wb") as f:
        pickle.dump(G_proj, f, protocol=pickle.HIGHEST_PROTOCOL)
    mb = out_path.stat().st_size / (1024 * 1024)
    print(f"[build_walking_graph] wrote {out_path}  ({mb:.1f} MB)",
          flush=True)


if __name__ == "__main__":
    main()
