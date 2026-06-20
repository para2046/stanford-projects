"""QC viz showing the 3 annotation variants side-by-side for the SAME
(frame, destination) pair.

Renders STUDENT input + TEACHER target output for each variant — these
are the (prompt, response) training pairs the LoRA will be fine-tuned
on. The teacher_prompt is shown only as a foldable diagnostic.

  python -m src.a2_viz_sft
"""

from __future__ import annotations

import argparse
import html
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
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--out", default="viz/a2_viz_sft.html")
    ap.add_argument("--strip-visible", action="store_true",
                    help="Show the V-stripped student prompts (matching "
                         "the SFT data when --strip-visible was used).")
    args = ap.parse_args()

    # Load the 3 single-row variant files
    annots = {}
    for var in ("given", "derived", "implicit"):
        p = config.CITY_DIR / "a2" / f"annotations_a2_{var}.jsonl"
        if not p.exists():
            print(f"[viz_sft] WARN: {p} missing")
            continue
        rows = [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]
        if rows:
            annots[var] = rows[0]
    print(f"[viz_sft] loaded variants: {list(annots)}")

    if "given" not in annots:
        print("[viz_sft] no 'given' annotation to anchor on, aborting")
        return

    # All 3 should be on the same (frame, destination) — anchor on 'given'
    base = annots["given"]
    key = (base["video"], base["frame_id"], base["destination"])

    # Look up route info for the map
    routes = {}
    for line in (config.CITY_DIR / "a2"
                 / "routes.jsonl").open(encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        routes[(r["video"], r["frame_id"], r["destination"])] = r
    r_route = routes.get(key)

    with (config.CITY_DIR / "osm_walking.pkl").open("rb") as f:
        G = pickle.load(f)
    to_latlon = _make_projector(G)

    if r_route:
        path = r_route["route_node_ids"]
        coords = [_node_latlon(G, int(n), to_latlon) for n in path]
        walker = r_route["current_gps_snapped"]
        dest = r_route["target_gps"]
        heading = r_route["heading"]
        arrow_len = 25.0
        h_rad = math.radians(heading)
        dlat = (arrow_len / 111000.0) * math.cos(h_rad)
        dlon = (arrow_len / (111000.0 * math.cos(math.radians(walker[0])))) \
               * math.sin(h_rad)
        arrow_end = [walker[0] + dlat, walker[1] + dlon]
        all_lats = [walker[0], dest[0]] + [c[0] for c in coords]
        all_lons = [walker[1], dest[1]] + [c[1] for c in coords]
        bounds = [[min(all_lats) - 0.0004, min(all_lons) - 0.0004],
                  [max(all_lats) + 0.0004, max(all_lons) + 0.0004]]
    else:
        coords, walker, dest, heading, arrow_end, bounds = (
            None, None, None, None, None, None)

    frame_path = config.FRAMES_DIR / base["video"] / f"{base['frame_id']}.jpg"

    css = """
    body { font: 13px -apple-system, Segoe UI, Helvetica, sans-serif;
           background: #f5f5f5; margin: 16px; }
    h1 { font-size: 19px; margin: 0 0 8px; }
    h2 { font-size: 15px; padding: 6px 12px; color: #fff;
         border-radius: 4px; margin: 16px 0 10px; }
    h2.given    { background: #2a9d8f; }
    h2.derived  { background: #f4a261; }
    h2.implicit { background: #e76f51; }
    .ban { font-size: 12px; color: #666; margin-bottom: 16px; }
    .grid-top { display: grid; grid-template-columns: 380px 1fr;
                gap: 16px; background: #fff; padding: 12px;
                border: 1px solid #ddd; border-radius: 8px;
                margin-bottom: 18px; }
    .grid-top img { width: 100%; height: auto; max-height: 240px;
                     object-fit: cover; border-radius: 4px;
                     outline: 3px solid #3a86ff; outline-offset: -3px; }
    #map { height: 300px; width: 100%; border-radius: 4px;
           border: 1px solid #ddd; }
    .info { font-size: 12px; line-height: 1.6; }
    .info table { border-collapse: collapse; width: 100%; }
    .info td { padding: 3px 8px; vertical-align: top; }
    .info td:first-child { color: #888; width: 40%; }
    section.variant { background: #fff; border: 1px solid #ddd;
                        padding: 16px; border-radius: 8px;
                        margin-bottom: 16px; }
    .row { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
    .col { font-size: 12px; }
    .col h3 { font-size: 13px; margin: 0 0 6px;
              color: #2c3e50; }
    .input  { background: #eef5fa; border-left: 4px solid #3a86ff;
              padding: 10px 12px; border-radius: 0 4px 4px 0;
              white-space: pre-wrap; font-family: ui-monospace,
              Menlo, Consolas, monospace; font-size: 11px; }
    .target { background: #fff3cd; border-left: 4px solid #f4a261;
              padding: 10px 12px; border-radius: 0 4px 4px 0;
              white-space: pre-wrap; font-family: ui-monospace,
              Menlo, Consolas, monospace; font-size: 11px; }
    .badge { display: inline-block; padding: 2px 8px;
             border-radius: 10px; font-size: 11px;
             font-weight: 600; margin-right: 4px; }
    .badge.ok  { background: #2a9d8f; color: #fff; }
    .badge.bad { background: #c0392b; color: #fff; }
    .verb { display: inline-block; padding: 4px 10px;
            background: #2c3e50; color: #fff; border-radius: 4px;
            font-weight: 600; font-size: 12px; }
    details { margin-top: 8px; }
    summary { cursor: pointer; color: #3a86ff; font-size: 11px; }
    pre.diag { background: #f5f5f5; padding: 8px;
                font-size: 10.5px; border-radius: 4px;
                white-space: pre-wrap; max-height: 320px;
                overflow-y: auto; }
    """

    out = ['<!DOCTYPE html><html><head><meta charset="utf-8">']
    out.append('<title>a2 SFT QC — 3 variants</title>')
    out.append('<link rel="stylesheet" '
                'href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">')
    out.append('<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js">'
                '</script>')
    out.append(f'<style>{css}</style></head><body>')
    out.append('<h1>SFT training pairs — 3 variants on the same frame</h1>')
    out.append('<div class="ban">Each variant shows the STUDENT PROMPT '
               '(blue, what the model sees at training and inference) and '
               'the TEACHER RESPONSE (orange, what the model is trained to '
               'emit). The teacher_prompt is foldable below — it is the '
               'labeller\'s shortcut input, never seen by the student.</div>')

    # Shared frame info + map at top
    out.append('<div class="grid-top">')
    out.append(f'<div><img src="{_file_url(frame_path)}">'
                '<div id="map" style="margin-top:8px;"></div></div>')
    out.append('<div class="info"><table>'
                f'<tr><td>frame</td><td>{base["video"]}/{base["frame_id"]}</td></tr>'
                f'<tr><td>destination</td><td>{base["destination"]} ({base["destination_zh"]})</td></tr>')
    if r_route:
        out.append(
            f'<tr><td>heading (true)</td><td><b>{r_route["heading"]:.0f}°</b></td></tr>'
            f'<tr><td>route 1st-seg bearing</td><td><b>{r_route["route_bearing_network"]:.0f}°</b></td></tr>'
            f'<tr><td>route distance</td><td>{r_route["route_distance_m"]:.0f} m</td></tr>'
            f'<tr><td>1st segment length</td><td>{r_route["first_segment_length_m"]:.0f} m</td></tr>'
            f'<tr><td># segments</td><td>{r_route["n_segments"]}</td></tr>'
            f'<tr><td>visible landmarks</td><td>{", ".join(base.get("visible_landmarks") or []) or "(none)"}</td></tr>'
            f'<tr><td>GT verb</td><td><span class="verb">{base["gt_verb"]}</span></td></tr>')
    out.append('</table></div></div>')

    # ── one section per variant ────────────────────────────────────
    for var in ("given", "derived", "implicit"):
        if var not in annots:
            continue
        a = annots[var]
        fmt = a["format_pass"]
        dirp = a["direction_pass"]
        fmt_b = f'<span class="badge {"ok" if fmt else "bad"}">format {"PASS" if fmt else "FAIL"}</span>'
        dir_b = f'<span class="badge {"ok" if dirp else "bad"}">direction {"PASS" if dirp else "FAIL"}</span>'
        verb_b = f'<span class="verb">{a["first_verb"]}</span> vs GT <span class="verb">{a["gt_verb"]}</span>'
        derived_h_line = ""
        if var == "derived":
            dh = a.get("derived_heading")
            if dh is not None:
                derived_h_line = (
                    f'<div style="margin-top:6px;font-size:12px;">'
                    f'<b>Heading derivation:</b> '
                    f'estimated <span class="verb">{dh:.0f}°</span> '
                    f'vs true <span class="verb">{a["heading"]:.0f}°</span>'
                    f' &mdash; '
                    f'{"✓ within 22.5°" if abs(((dh-a["heading"]+180)%360)-180) < 22.5 else "✗ off"}'
                    f'</div>')

        out.append(f'<section class="variant">')
        out.append(f'<h2 class="{var}">{var.upper()}  &middot;  '
                    f'{fmt_b} {dir_b}  &middot;  verb: {verb_b}</h2>')
        if derived_h_line:
            out.append(derived_h_line)

        out.append('<div class="row">')
        # If --strip-visible, hide the "Visible landmarks at this spot:" block
        # from the displayed student_prompt (matching what the SFT pipeline does).
        display_prompt = a["student_prompt"]
        if args.strip_visible:
            import re as _re
            display_prompt = _re.sub(
                r"Visible landmarks at this spot:\n\s+[^\n]+\n\n", "",
                display_prompt, flags=_re.MULTILINE)
        out.append(
            f'<div class="col"><h3>STUDENT PROMPT (input — what the model sees)</h3>'
            f'<div class="input">{html.escape(display_prompt)}</div></div>')
        out.append(
            f'<div class="col"><h3>TEACHER RESPONSE (target — what the model is trained to emit)</h3>'
            f'<div class="target">{html.escape(a["response"])}</div></div>')
        out.append('</div>')

        out.append(
            f'<details>'
            f'<summary>diagnostic: teacher_prompt (sent to Gemini, NOT seen by student)</summary>'
            f'<pre class="diag">{html.escape(a["teacher_prompt"])}</pre>'
            f'</details>')
        out.append('</section>')

    out.append('</body></html>')

    # Leaflet map script
    if r_route:
        out.append(f'''<script>
(function() {{
  const map = L.map("map").fitBounds({json.dumps(bounds)});
  L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png',
              {{maxZoom: 19, attribution: '© OpenStreetMap'}}).addTo(map);
  L.circleMarker([{walker[0]}, {walker[1]}], {{
    radius: 8, color: "#1f77b4", fillColor: "#1f77b4", fillOpacity: 0.7
  }}).addTo(map).bindPopup("WALKER · heading {heading:.0f}°");
  L.polyline([[{walker[0]}, {walker[1]}], [{arrow_end[0]}, {arrow_end[1]}]],
             {{color: "#1f77b4", weight: 4, opacity: 0.9}}).addTo(map);
  L.polyline({json.dumps([list(c) for c in coords])},
             {{color: "#d62728", weight: 5, opacity: 0.7,
              dashArray: '6,4'}}).addTo(map)
    .bindPopup("OSM route — {r_route['n_segments']} segment(s)");
  L.marker([{dest[0]}, {dest[1]}]).addTo(map)
    .bindPopup("DESTINATION · {base['destination']}");
}})();
</script>''')

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(out), encoding="utf-8")
    print(f"[viz_sft] wrote {out_path.resolve()}")
    print(f"           open: {_file_url(out_path)}")


if __name__ == "__main__":
    main()
