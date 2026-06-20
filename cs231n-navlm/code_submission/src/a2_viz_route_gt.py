"""Pick 3 unambiguous + 1 boundary example from routes.jsonl, render
to viz/a2_route_gt.html as Leaflet maps with the OSM route drawn.

For each example:
  - QUERY frame image (top-left)
  - Leaflet map (right):
      blue arrow  = walker position + heading
      red line    = OSM shortest path (the route)
      green pin   = destination
  - Info table:
      heading, edge_bearing, route distance, all 4 verb errors,
      gt_verb, second-best verb (if close)

Pick rules:
  FIXED (3) :  best_error <  10° AND second_best > 60°
  MULTI (1) :  two verbs within 30° of each other (boundary case)
"""

from __future__ import annotations

import json
import math
import pickle
import sys
from pathlib import Path
from urllib.parse import quote

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config                                       # noqa: E402


def _file_url(p):
    return "file:///" + quote(str(Path(p).resolve()).replace("\\", "/"))


def _make_projector(G):
    import pyproj
    crs = G.graph.get("crs")
    if not crs or "4326" in str(crs):
        return None
    return pyproj.Transformer.from_crs(crs, "EPSG:4326", always_xy=True)


def _node_latlon(G, node_id, to_latlon):
    if to_latlon:
        lon, lat = to_latlon.transform(G.nodes[node_id]["x"],
                                        G.nodes[node_id]["y"])
        return lat, lon
    return G.nodes[node_id]["y"], G.nodes[node_id]["x"]


def main():
    # ── load ───────────────────────────────────────────────────────
    routes = []
    for line in (config.CITY_DIR / "a2"
                 / "routes.jsonl").open(encoding="utf-8"):
        if not line.strip():
            continue
        routes.append(json.loads(line))
    print(f"[viz_route_gt] routes total: {len(routes):,}")

    with (config.CITY_DIR / "osm_walking.pkl").open("rb") as f:
        G = pickle.load(f)
    to_latlon = _make_projector(G)

    # ── pick examples ──────────────────────────────────────────────
    def sort_verbs(errors):
        return sorted(errors.items(), key=lambda kv: kv[1])

    fixed_pool = []
    multi_pool = []
    for r in routes:
        if r["n_segments"] < 1 or not r.get("verb_errors"):
            continue
        sorted_v = sort_verbs(r["verb_errors"])
        best_err = sorted_v[0][1]
        second_err = sorted_v[1][1] if len(sorted_v) >= 2 else 180.0
        if best_err < 10 and second_err > 60:
            fixed_pool.append(r)
        # multi-answer: route bearing near a verb-boundary (~45°/135°/...),
        # so best and second-best verbs are both moderately wrong but
        # close to each other (within 15°).
        elif (best_err < 50 and second_err < 60
              and (second_err - best_err) < 15):
            multi_pool.append(r)

    print(f"[viz_route_gt] FIXED candidates: {len(fixed_pool)}")
    print(f"[viz_route_gt] MULTI candidates: {len(multi_pool)}")

    # pick 3 fixed with VARIED verbs
    fixed_examples = []
    picked_verbs = set()
    for r in fixed_pool:
        v = r["gt_verb"]
        if v in picked_verbs:
            continue
        fixed_examples.append(r)
        picked_verbs.add(v)
        if len(fixed_examples) >= 3:
            break
    # fall back if not 3 distinct verbs
    for r in fixed_pool:
        if len(fixed_examples) >= 3:
            break
        if r not in fixed_examples:
            fixed_examples.append(r)

    multi_examples = multi_pool[:1] if multi_pool else []

    print(f"[viz_route_gt] picked: {len(fixed_examples)} FIXED + "
          f"{len(multi_examples)} MULTI")

    # ── helper for the per-example map ─────────────────────────────
    def make_map_block(r, idx, label):
        # collect route polyline lat/lon
        path = r["route_node_ids"]
        coords = [_node_latlon(G, int(n), to_latlon) for n in path]
        # walker position
        walker = r["current_gps_snapped"]
        # destination
        dest = r["target_gps"]
        heading = r["heading"]
        # heading arrow end (~20m to the north visually scaled)
        arrow_len_m = 25.0
        h_rad = math.radians(heading)
        # rough: 1° latitude ≈ 111 km, so 25m ≈ 0.000225°
        dlat = (arrow_len_m / 111000.0) * math.cos(h_rad)
        dlon = (arrow_len_m / (111000.0 * math.cos(math.radians(walker[0])))) * math.sin(h_rad)
        arrow_end = [walker[0] + dlat, walker[1] + dlon]

        # frame image
        frame_path = config.FRAMES_DIR / r["video"] / f"{r['frame_id']}.jpg"

        # info table
        errors = r["verb_errors"]
        sorted_v = sort_verbs(errors)
        verb_rows = ""
        for vrb, err in sorted_v:
            mark = ' style="background:#fff3cd; font-weight:600;"' if vrb == r["gt_verb"] else ''
            verb_rows += (
                f'<tr{mark}><td>{vrb}</td>'
                f'<td>{err:.1f}°</td></tr>')

        map_id = f"map{idx}"
        # bounds: include walker + dest + all route nodes
        all_lats = [walker[0], dest[0]] + [c[0] for c in coords]
        all_lons = [walker[1], dest[1]] + [c[1] for c in coords]
        bounds = ((min(all_lats) - 0.0005, min(all_lons) - 0.0005),
                  (max(all_lats) + 0.0005, max(all_lons) + 0.0005))

        return f'''
<section class="example {'fixed' if 'FIXED' in label else 'multi'}">
  <h2>{label}: {r["video"]}/{r["frame_id"]} → {r["destination"]} ({r["destination_zh"]})</h2>
  <div class="grid">
    <div class="left">
      <img class="qframe" src="{_file_url(frame_path)}" loading="lazy">
      <table class="info">
        <tr><td>Heading</td><td><b>{heading:.0f}°</b></td></tr>
        <tr><td>Route bearing (first edge)</td>
            <td><b>{r["route_bearing_network"]:.0f}°</b></td></tr>
        <tr><td>Route distance</td>
            <td>{r["route_distance_m"]:.0f} m</td></tr>
        <tr><td>First segment length</td>
            <td>{r["first_segment_length_m"]:.0f} m</td></tr>
        <tr><td>Sampling band</td><td>{r["sampling_band"]}</td></tr>
        <tr><td>Destination kind</td><td>{r["destination_kind"]}</td></tr>
      </table>
      <h3>Verb errors (gap from new_heading to route_bearing)</h3>
      <table class="verbs">
        <tr><th>verb</th><th>error</th></tr>
        {verb_rows}
      </table>
      <div class="note">GT verb = <b>{r["gt_verb"]}</b>
       (error {r["verb_error_deg"]:.0f}°)</div>
    </div>
    <div class="right">
      <div id="{map_id}" class="map"></div>
    </div>
  </div>
</section>
<script>
(function() {{
  const map = L.map("{map_id}").fitBounds([[{bounds[0][0]},{bounds[0][1]}],
                                            [{bounds[1][0]},{bounds[1][1]}]]);
  L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
      maxZoom: 19,
      attribution: '© OpenStreetMap'
  }}).addTo(map);
  // walker position
  L.circleMarker([{walker[0]}, {walker[1]}], {{
      radius: 8, color: "#1f77b4", fillColor: "#1f77b4", fillOpacity: 0.7
  }}).addTo(map).bindPopup("WALKER · heading {heading:.0f}°");
  // heading arrow
  L.polyline([[{walker[0]}, {walker[1]}], [{arrow_end[0]}, {arrow_end[1]}]],
             {{color: "#1f77b4", weight: 4, opacity: 0.9}}).addTo(map);
  // route polyline
  L.polyline({json.dumps([list(c) for c in coords])},
             {{color: "#d62728", weight: 5, opacity: 0.7,
              dashArray: '6,4'}}).addTo(map)
    .bindPopup("OSM shortest path — {r['n_segments']} segment(s)");
  // destination
  L.marker([{dest[0]}, {dest[1]}]).addTo(map)
    .bindPopup("DESTINATION · {r['destination']}");
}})();
</script>
'''

    css = """
    body { font: 13px -apple-system, Segoe UI, Helvetica, sans-serif;
           background: #f5f5f5; margin: 16px; }
    h1 { font-size: 18px; }
    section.example { background: #fff; border: 1px solid #ddd;
                       padding: 16px; border-radius: 8px;
                       margin-bottom: 20px; }
    section.fixed h2 { color: #2a9d8f; }
    section.multi h2 { color: #e76f51; }
    .grid { display: grid; grid-template-columns: 360px 1fr; gap: 16px; }
    .left img.qframe { width: 100%; height: auto; max-height: 240px;
                        object-fit: cover; border-radius: 4px;
                        outline: 3px solid #3a86ff; outline-offset: -3px; }
    table { border-collapse: collapse; width: 100%; margin-top: 8px;
            font-size: 12px; }
    table td, table th { padding: 4px 8px; border-bottom: 1px solid #eee;
                          text-align: left; }
    table.info td:first-child { color: #888; width: 50%; }
    table.verbs th { background: #f0f0f0; }
    .note { margin-top: 8px; font-size: 12px; color: #666; }
    .map { height: 380px; width: 100%; border-radius: 4px;
           border: 1px solid #ddd; }
    h3 { font-size: 13px; margin: 10px 0 4px; }
    """

    out = ['<!DOCTYPE html><html><head><meta charset="utf-8">']
    out.append('<title>a2 route GT examples</title>')
    out.append('<link rel="stylesheet" '
                'href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">')
    out.append('<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js">'
                '</script>')
    out.append(f'<style>{css}</style></head><body>')
    out.append('<h1>a2_route GT verbs — 3 fixed + 1 multi-answer example</h1>')
    out.append('<div style="color:#666;font-size:12px;margin-bottom:14px;">'
               'For each row: walker (blue circle + arrow = heading), '
               'OSM shortest path (dashed red), destination (green pin). '
               'The verb-errors table shows the gap from <code>(heading + '
               'ACTION_DELTA[verb]) mod 360</code> to '
               '<code>first_edge_bearing</code>; the smallest gap wins.'
               '</div>')

    for i, r in enumerate(fixed_examples, 1):
        out.append(make_map_block(r, i, f"FIXED-{i}"))
    for i, r in enumerate(multi_examples, 1):
        out.append(make_map_block(r, 100 + i, f"MULTI-{i} (boundary)"))

    out.append('</body></html>')
    out_path = Path("viz/a2_route_gt.html")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(out), encoding="utf-8")
    print(f"[viz_route_gt] wrote {out_path.resolve()}")
    print(f"             open: {_file_url(out_path)}")


if __name__ == "__main__":
    main()
