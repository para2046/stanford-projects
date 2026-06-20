"""Generate docs/figures/fig5_map_vlm_grounding.png — 2x2 grid showing
how 'map routes + VLM visual grounding' can succeed OR fail.

  row 1 = SUCCESS  bahnhofstrasse/frame_00276 → Limmat river
            walker faces 0° (north), route bears 89° (east) → 'turn right'
            VLM agrees with map.
  row 2 = FAILURE  looks_perfect/frame_02681 → Lake Zurich
            walker is in a narrow Zurich old-town alley (cobblestones,
            outdoor cafés, no water in view), facing 90° (east),
            route bears 189° (south, 500 m) → 'turn right'.
            VLM predicted 'turn around' — the lake is 500 m away and
            invisible from the alley, so the model cannot visually
            ground the destination. This failure repeats across 17 of
            our trained conditions.

Layout per row:
  col 1 = walker's view (the photo the VLM sees)
  col 2 = OSM map view — walking-graph context + walker pose + route polyline
            + destination star
"""
import io
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config                                       # noqa: E402

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# ── two examples (success / failure for direction) ─────────────────────
EXAMPLES = [
    {
        "row": 0, "label": "SUCCESS", "color": "tab:green",
        "video": "bahnhofstrasse", "frame_id": "frame_00276",
        "dest": "Limmat river",
        "heading": 0.0, "route_bearing": 89.5,
        "gt_verb": "turn right", "pred_verb": "turn right",
        "answer": ('"Notice Bahnhofstrasse with its shops and tram tracks\n'
                    'stretching ahead of you.    Turn right."'),
        "route_node_ids": [9657081383, 9656216327, 9633032363, 9633070913,
                            5888716828, 5888716830, 10820680734, 10820680735,
                            1600187449],
        "walker_latlon": (47.3738173, 8.5381547),
        "target_latlon": (47.3738439, 8.5418601),
        "distance_m": 327.1,
    },
    {
        "row": 1, "label": "FAILURE", "color": "tab:red",
        "video": "looks_perfect", "frame_id": "frame_02681",
        "dest": "Lake Zurich",
        "heading": 90.0, "route_bearing": 189.4,
        "gt_verb": "turn right", "pred_verb": "turn around",
        "answer": ('"Notice how Niederdorfstrasse continues ahead of you.\n'
                    '    Turn around."'),
        "route_node_ids": [660831174, 660831175, 2500522126, 2500446884,
                            5715280504, 5715280505, 5715288070, 6576627816,
                            6576627817, 6576627818, 5888735647, 5888735648,
                            5888735649, 5888735650, 5888735651, 5888735652,
                            6635185175, 5888784625, 6635185174, 5888739561,
                            5888768412, 5888739562, 5888739565, 5888739566,
                            5888768408, 5888768409, 5888768410, 5888768411],
        "walker_latlon": (47.374741100000016, 8.5432386),
        "target_latlon": (47.3703151, 8.5432493),
        "distance_m": 499.5,
    },
]

# ── load OSM walking graph + lat/lon transformer (once) ────────────────
print("[fig5] loading osm_walking.pkl ...")
with (config.CITY_DIR / "osm_walking.pkl").open("rb") as f:
    G = pickle.load(f)
print(f"[fig5] graph: {G.number_of_nodes():,} nodes / {G.number_of_edges():,} edges")

from pyproj import Transformer    # noqa: E402
crs = G.graph.get("crs")
to_latlon = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)

def node_latlon(n):
    nd = G.nodes[n]
    lon, lat = to_latlon.transform(nd["x"], nd["y"])
    return lat, lon

# ── compose figure (2 rows x 2 cols) ───────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 11), dpi=300,
                          gridspec_kw={"width_ratios": [1.78, 1.0]})

for ex in EXAMPLES:
    row = ex["row"]
    # ── compute route polyline + bbox ─────────────────────────────────
    route_pts = [node_latlon(n) for n in ex["route_node_ids"]]
    lats = [p[0] for p in route_pts] + [ex["walker_latlon"][0], ex["target_latlon"][0]]
    lons = [p[1] for p in route_pts] + [ex["walker_latlon"][1], ex["target_latlon"][1]]
    lat_min, lat_max = min(lats), max(lats)
    lon_min, lon_max = min(lons), max(lons)
    margin_lat = (lat_max - lat_min) * 0.35 + 0.0005
    margin_lon = (lon_max - lon_min) * 0.20 + 0.0005

    # ── nearby walking-graph edges ────────────────────────────────────
    lat_lo, lat_hi = lat_min - margin_lat, lat_max + margin_lat
    lon_lo, lon_hi = lon_min - margin_lon, lon_max + margin_lon
    nearby_edges = []
    for u, v, _ in G.edges(data=True):
        u_lat, u_lon = node_latlon(u)
        if not (lat_lo <= u_lat <= lat_hi and lon_lo <= u_lon <= lon_hi):
            continue
        v_lat, v_lon = node_latlon(v)
        if not (lat_lo <= v_lat <= lat_hi and lon_lo <= v_lon <= lon_hi):
            continue
        nearby_edges.append(((u_lon, u_lat), (v_lon, v_lat)))
    print(f"[fig5] row{row} viewport edges: {len(nearby_edges):,}")

    # ── LEFT: walker's view ──────────────────────────────────────────
    ax = axes[row, 0]
    img = Image.open(config.FRAMES_DIR / ex["video"] / f'{ex["frame_id"]}.jpg').convert("RGB")
    ax.imshow(img)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)
    ax.text(0.02, 0.97, ex["label"], transform=ax.transAxes,
            fontsize=22, fontweight="bold", color="white", va="top",
            bbox=dict(facecolor=ex["color"], edgecolor="none", pad=6))
    ax.text(0.98, 0.97, f'{ex["video"]}/{ex["frame_id"]}',
            transform=ax.transAxes, fontsize=14, color="white",
            va="top", ha="right",
            bbox=dict(facecolor="black", edgecolor="none", pad=4, alpha=0.6))
    pred_color = "tab:green" if ex["pred_verb"] == ex["gt_verb"] else "tab:red"
    ax.set_xlabel(
        f'{ex["answer"]}\n'
        f'GT verb (from map): "{ex["gt_verb"]}"   |   '
        f'VLM predicted: "{ex["pred_verb"]}"',
        fontsize=13, color=pred_color, fontweight="bold", loc="left")

    # ── RIGHT: OSM map view ──────────────────────────────────────────
    ax = axes[row, 1]
    for (a, b) in nearby_edges:
        ax.plot([a[0], b[0]], [a[1], b[1]],
                color="#cccccc", linewidth=0.8, zorder=1)
    rlons = [p[1] for p in route_pts]
    rlats = [p[0] for p in route_pts]
    ax.plot(rlons, rlats, color="#d62728", linewidth=2.8, linestyle="--",
            zorder=4)
    # walker dot + heading arrow
    wlon, wlat = ex["walker_latlon"][1], ex["walker_latlon"][0]
    ax.plot(wlon, wlat, "o", color="tab:blue", markersize=14, zorder=6)
    arrow_len_lat = (lat_max - lat_min + 2 * margin_lat) * 0.18
    h_rad = np.deg2rad(ex["heading"])     # 0° = north, +cw
    dy = arrow_len_lat * np.cos(h_rad)
    dx = arrow_len_lat * np.sin(h_rad)    # lon scale ≈ lat at 47° / cos(47°)
    dx = dx / np.cos(np.deg2rad(wlat))
    ax.annotate("", xy=(wlon + dx, wlat + dy),
                 xytext=(wlon, wlat),
                 arrowprops=dict(arrowstyle="->", color="tab:blue",
                                  linewidth=3), zorder=7)
    # destination star
    tlon, tlat = ex["target_latlon"][1], ex["target_latlon"][0]
    ax.plot(tlon, tlat, "*", color="tab:green", markersize=24,
            markeredgecolor="white", markeredgewidth=1.5, zorder=6)
    ax.text(tlon, tlat, f"  {ex['dest']}", fontsize=14, color="tab:green",
             fontweight="bold", va="center", ha="left", zorder=8,
             bbox=dict(facecolor="white", edgecolor="none",
                        alpha=0.7, pad=1.5))
    ax.set_xlim(lon_min - margin_lon, lon_max + margin_lon)
    ax.set_ylim(lat_min - margin_lat, lat_max + margin_lat)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)
    ax.set_xlabel(
        f'walker heading {ex["heading"]:.0f}°  ·  '
        f'route bears {ex["route_bearing"]:.0f}°  ·  '
        f'{ex["distance_m"]:.0f} m',
        fontsize=14)
    if row == 0:
        ax.set_title("OSM walking-graph route",
                      fontsize=18, loc="center")

# legend (manual, once) — placed at the bottom outside the grid
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
legend_handles = [
    mlines.Line2D([], [], color="tab:blue", marker="o", linestyle="",
                   markersize=10, label="walker position"),
    mlines.Line2D([], [], color="tab:blue", marker=">", linestyle="-",
                   linewidth=2, markersize=8, label="walker heading"),
    mlines.Line2D([], [], color="#d62728", linewidth=2.5, linestyle="--",
                   label="OSM-routed path"),
    mlines.Line2D([], [], color="tab:green", marker="*", linestyle="",
                   markersize=15, label="destination"),
    mlines.Line2D([], [], color="#cccccc", linewidth=1,
                   label="walking-graph edges"),
]
fig.legend(handles=legend_handles, loc="lower center", ncol=5,
            fontsize=13, frameon=False, bbox_to_anchor=(0.5, 0.0))

plt.subplots_adjust(left=0.04, right=0.99, top=0.97, bottom=0.06,
                     wspace=0.05, hspace=0.06)
out = Path("docs/figures/fig5_map_vlm_grounding.png")
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, bbox_inches="tight")
plt.close()
print(f"saved {out}")
