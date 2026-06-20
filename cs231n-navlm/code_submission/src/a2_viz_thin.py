"""Render ALL matched frames for the thin-cohort attractions, grouped
by attraction, so we can visually decide whether to keep / drop / augment
each one.

Targets:
  ≤1 frame  : Kunsthaus, Bürkliplatz, Paradeplatz
  11-30 frm : Helmhaus, Sechseläutenplatz, Opernhaus, Landesmuseum

Each attraction gets a section. Within a section, each frame is one row:
  QUERY frame · 4 compass crops at top-1 pano · GPS / VLM info · matches.

Outputs viz/a2_thin_attractions.html.

  python -m src.a2_viz_thin
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path
from urllib.parse import quote

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config                                       # noqa: E402
from src.a2_attraction_slots import ATTRACTIONS_21  # noqa: E402


THIN_ATTRACTIONS = [
    ("Kunsthaus",         "≤1 frame"),
    ("Bürkliplatz",       "≤1 frame"),
    ("Paradeplatz",       "≤1 frame"),
    ("Helmhaus",          "11-30"),
    ("Sechseläutenplatz", "11-30"),
    ("Opernhaus",         "11-30"),
    ("Landesmuseum",      "11-30"),
]

CANON = {en for en, *_ in ATTRACTIONS_21}
ATTR_META = {en: (zh, kind)
             for en, zh, _lat, _lon, kind in ATTRACTIONS_21}


def _file_url(p):
    return "file:///" + quote(str(Path(p).resolve()).replace("\\", "/"))


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
    # load matched cohort
    matched = []
    for line in (config.CITY_DIR / "a2"
                 / "GPS_VLM_GEO.jsonl").open(encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("matched"):
            matched.append(r)

    # load gps_recovery for top_sv_id + s_dino + heading
    gps_meta = {}
    for line in (config.CITY_DIR
                 / "gps_recovery_full.jsonl").open(encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("top_sv_id"):
            gps_meta[(r["video"], r["frame_id"])] = {
                "top_sv_id": r["top_sv_id"],
                "s_dino": r.get("s_dino", 0.0),
                "heading": r.get("heading") or 0,
            }

    # load SV meta
    sv_meta = {}
    for line in (config.STREETVIEW_DIR
                 / "meta.jsonl").open(encoding="utf-8"):
        if not line.strip():
            continue
        m = json.loads(line)
        sv_meta[m["id"]] = m

    # heading_v2
    heading_v2 = {}
    hv2 = config.CITY_DIR / "a2" / "heading_v2.jsonl"
    if hv2.exists():
        for line in hv2.open(encoding="utf-8"):
            if not line.strip():
                continue
            d = json.loads(line)
            heading_v2[(d["video"], d["frame_id"])] = d

    # DINOv2 embeddings for the 4-crop cosine
    cdir = config.CITY_DIR / "dinov2"
    sv_cache = np.load(cdir / "sv_v1.npz", allow_pickle=True)
    fr_cache = np.load(cdir / "frames_n1_l0.npz", allow_pickle=True)
    sv_embs = sv_cache["embs"]
    sv_ids = [Path(str(s)).stem for s in sv_cache["paths"]]
    fr_embs = fr_cache["embs"]
    fr_paths = [Path(p) for p in fr_cache["paths"]]
    fr_idx = {(p.parent.name, p.stem): i for i, p in enumerate(fr_paths)}
    pano_to_sv = collections.defaultdict(list)
    for j, sid in enumerate(sv_ids):
        pid = sv_meta.get(sid, {}).get("pano_id", "")
        if pid:
            pano_to_sv[pid].append(j)

    # group matched frames by attraction
    by_attr = collections.defaultdict(list)
    for r in matched:
        for en in frame_attractions(r):
            by_attr[en].append(r)

    # ── render ──────────────────────────────────────────────────────
    css = """
    body { font: 13px -apple-system, Segoe UI, Helvetica, sans-serif;
           background: #f5f5f5; margin: 16px; }
    h1 { font-size: 18px; margin: 0 0 12px; }
    h2 { font-size: 16px; margin: 24px 0 12px; padding: 8px 12px;
         background: #2c3e50; color: #fff; border-radius: 6px; }
    .ban { font-size: 12px; color: #888; margin-bottom: 16px; }
    .row { display: grid;
           grid-template-columns: 200px repeat(4, 130px) 1fr;
           gap: 8px; padding: 12px; background: #fff;
           border: 1px solid #ddd; border-radius: 6px;
           margin-bottom: 10px; }
    .cell img { width: 100%; height: 100px; object-fit: cover;
                border-radius: 4px; }
    .cell.q img { height: 150px; }
    .cell.q img { outline: 3px solid #3a86ff; outline-offset: -3px; }
    .label { font-size: 11px; color: #555; margin-top: 4px; line-height: 1.4; }
    .info { font-size: 12px; line-height: 1.5; }
    .info table { border-collapse: collapse; width: 100%; }
    .info td { padding: 2px 6px; vertical-align: top; }
    .info td:first-child { color: #888; font-size: 11px;
                            white-space: nowrap; width: 80px; }
    .badge { display: inline-block; padding: 2px 8px; border-radius: 10px;
             font-size: 11px; font-weight: 600; }
    .badge.attraction { background: #2a9d8f; color: #fff; }
    .badge.landmark { background: #e9c46a; color: #333; }
    .badge.poi { background: #ccc; color: #333; }
    .badge.exact { background: #2a9d8f; color: #fff; }
    .badge.substring { background: #f4a261; color: #333; }
    .badge.word_share { background: #e76f51; color: #fff; }
    .match-row { font-family: ui-monospace, Menlo, Consolas, monospace;
                 font-size: 11px; margin: 2px 0; }
    .name { font-weight: 600; }
    .good { color: #2a9d8f; font-weight: 600; }
    .ok   { color: #e9c46a; font-weight: 600; }
    .bad  { color: #c0392b; font-weight: 600; }
    .tier { display: inline-block; padding: 2px 8px; border-radius: 10px;
            font-size: 11px; font-weight: 600; background: #c0392b; color: #fff; }
    .tier.ok { background: #e9c46a; color: #333; }
    """

    html = ['<!DOCTYPE html><html><head><meta charset="utf-8">']
    html.append('<title>a2 — thin attractions (visual QC)</title>')
    html.append(f'<style>{css}</style></head><body>')
    html.append('<h1>Thin attractions — visual QC</h1>')
    html.append('<div class="ban">All matched frames per attraction. '
                'Use to decide drop / accept / augment per attraction.</div>')

    for attr_name, tier in THIN_ATTRACTIONS:
        frames = by_attr.get(attr_name, [])
        zh, kind = ATTR_META.get(attr_name, ("", ""))
        tier_class = "tier ok" if tier == "11-30" else "tier"
        html.append(f'<h2>{attr_name} ({zh}) · kind={kind} · '
                    f'<span class="{tier_class}">{tier}</span> · '
                    f'{len(frames)} frame(s)</h2>')
        if not frames:
            html.append('<div class="ban">(no matched frames)</div>')
            continue

        for r in frames:
            video, fid = r["video"], r["frame_id"]
            frame_path = config.FRAMES_DIR / video / f"{fid}.jpg"
            meta = gps_meta.get((video, fid), {})
            top_sv = meta.get("top_sv_id", "")
            s_dino = meta.get("s_dino", 0.0)
            heading = meta.get("heading", 0)

            # 4-crop cosines at top-1's pano
            crops = []
            if (video, fid) in fr_idx and top_sv:
                sims = sv_embs @ fr_embs[fr_idx[(video, fid)]]
                top_pano = sv_meta.get(top_sv, {}).get("pano_id", "")
                for j in pano_to_sv.get(top_pano, []):
                    crops.append((
                        sv_meta[sv_ids[j]].get("compass_angle", 0),
                        float(sims[j]),
                        sv_ids[j]))
                crops.sort(key=lambda t: t[0])
            best_cos = max((c for _, c, _ in crops), default=0.0)

            ga = r["list_a_gps"].get("attractions", [])
            gl = r["list_a_gps"].get("landmarks", [])
            gp = r["list_a_gps"].get("pois", [])
            va = r["list_b_vlm"].get("attractions", [])
            vl = r["list_b_vlm"].get("landmarks", [])
            vp = r["list_b_vlm"].get("pois", [])

            best = r.get("best_level") or ""
            best_badge = f'<span class="badge {best}">{best}</span>'

            hv2 = heading_v2.get((video, fid), {})
            h_v2 = hv2.get("heading_v2")
            h_v2_dec = hv2.get("decision", "")
            h_v2_gap = hv2.get("gap")
            h_v2_str = (f'{h_v2:.0f}°' if h_v2 is not None
                        else '<span class="bad">N/A</span>')
            dec_color = {"top1": "good", "top1+top2": "ok",
                         "ambiguous": "bad"}.get(h_v2_dec, "")
            dec_str = (f'<span class="{dec_color}">{h_v2_dec}</span>'
                       f' (gap {h_v2_gap:.2f})' if h_v2_dec else '—')

            match_html = []
            for m in r["matches"][:10]:
                mt = m["match_type"].split(":")[0]
                # highlight if this match involves the target attraction
                hl = (' style="background: #fff3cd; padding: 2px 4px;"'
                      if attr_name in (m["gps_name"], m["vlm_name"])
                      else '')
                match_html.append(
                    f'<div class="match-row"{hl}>'
                    f'<span class="badge {mt}">{mt}</span> '
                    f'GPS=<span class="name">{m["gps_name"]}</span> '
                    f'<span class="badge {m["gps_level"]}">{m["gps_level"]}</span>'
                    f' &nbsp;⇄&nbsp; '
                    f'VLM=<span class="name">{m["vlm_name"]}</span> '
                    f'<span class="badge {m["vlm_level"]}">{m["vlm_level"]}</span>'
                    f'</div>')
            if len(r["matches"]) > 10:
                match_html.append(
                    f'<div class="match-row">(+{len(r["matches"])-10} more)</div>')

            html.append('<div class="row">')
            html.append(
                f'<div class="cell q">'
                f'<img src="{_file_url(frame_path)}" loading="lazy">'
                f'<div class="label"><b>QUERY</b><br>{video}/{fid}<br>'
                f'h v1: <b>{heading:.0f}°</b>  h v2: <b>{h_v2_str}</b><br>'
                f'decision: {dec_str}<br>'
                f'best={best_badge}</div>'
                f'</div>')
            for h_deg, cos, sid in crops:
                sv_path = config.STREETVIEW_DIR / "images" / f"{sid}.jpg"
                is_best = abs(cos - best_cos) < 1e-9
                cls = ("good" if cos > 0.75
                       else ("ok" if cos >= 0.60 else "bad"))
                style = (' style="outline: 3px solid #d62728; '
                         'outline-offset: -3px;"' if is_best else '')
                star = " ★" if is_best else ""
                html.append(
                    f'<div class="cell">'
                    f'<img src="{_file_url(sv_path)}" loading="lazy"{style}>'
                    f'<div class="label">h <b>{int(h_deg):03d}°</b>{star}<br>'
                    f'<span class="{cls}">cos {cos:.3f}</span></div>'
                    f'</div>')
            for _ in range(max(0, 4 - len(crops))):
                html.append('<div class="cell"><div class="label">—</div></div>')

            info = ['<div class="info"><table>']
            info.append(f'<tr><td>GPS attr</td><td>{", ".join(ga) or "—"}</td></tr>')
            info.append(f'<tr><td>GPS land</td><td>{", ".join(gl[:5]) or "—"}'
                        f'{" …" if len(gl)>5 else ""}</td></tr>')
            info.append(f'<tr><td>VLM attr</td><td>{", ".join(va) or "—"}</td></tr>')
            info.append(f'<tr><td>VLM land</td><td>{", ".join(vl[:5]) or "—"}'
                        f'{" …" if len(vl)>5 else ""}</td></tr>')
            info.append(f'<tr><td>VLM pois</td><td>{", ".join(vp[:5]) or "—"}'
                        f'{" …" if len(vp)>5 else ""}</td></tr>')
            info.append('</table>')
            info.append('<div style="margin-top:6px;"><b>matches:</b></div>')
            info.extend(match_html)
            info.append('</div>')
            html.append("".join(info))
            html.append('</div>')

    html.append('</body></html>')

    out_path = Path("viz/a2_thin_attractions.html")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(html), encoding="utf-8")
    print(f"[viz_thin] wrote {out_path.resolve()}")
    print(f"           open: {_file_url(out_path)}")

    print()
    print("=== rendered frames per attraction ===")
    for attr_name, tier in THIN_ATTRACTIONS:
        n = len(by_attr.get(attr_name, []))
        print(f"  {attr_name:<22s} ({tier:<10s})  {n} frame(s)")


if __name__ == "__main__":
    main()
