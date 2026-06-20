"""STEP 4 — HTML grid of matched frames from GPS_VLM_GEO.jsonl.

Random samples N matched frames (default 30) and renders them
side-by-side with:
  - the QUERY video frame (blue border)
  - the top-1 Street View crop DINOv2 matched it to
  - GPS-side info: dino_nearest, attractions/landmarks in radius
  - VLM-side info: raw visible[], guess, canonical hits
  - the list of (gps_name ⇄ vlm_name, match_type) coincidences
  - a prominent best_level badge

For visually verifying that the matched cohort is actually correct
(no DINOv2 lookalike false-positives surviving the cos>=0.75 + VLM
agreement gate).

  python -m src.a2_viz_matched                          # 30 random
  python -m src.a2_viz_matched --n 50 --seed 7
  python -m src.a2_viz_matched --level attraction       # only attraction-level matches
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path
from urllib.parse import quote

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config                                       # noqa: E402


def _file_url(p):
    return "file:///" + quote(str(Path(p).resolve()).replace("\\", "/"))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--matched",
                    default=str(config.CITY_DIR / "a2"
                                / "GPS_VLM_GEO.jsonl"))
    ap.add_argument("--gps-recovery",
                    default=str(config.CITY_DIR
                                / "gps_recovery_full.jsonl"))
    ap.add_argument("--sv-meta",
                    default=str(config.STREETVIEW_DIR / "meta.jsonl"))
    ap.add_argument("--out", default="viz/a2_vlmagreed.html")
    ap.add_argument("--n", type=int, default=30,
                    help="random sample size (default 30)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--level", default=None,
                    choices=["attraction", "landmark", "poi"],
                    help="restrict sample to one best_level")
    ap.add_argument("--frame-cache", default="frames_n1_l0",
                    help="DINOv2 frame embedding cache name "
                         "(under data/cities/zurich/dinov2/)")
    args = ap.parse_args()

    # ── load matched rows + filter ──────────────────────────────────
    rows = []
    for line in Path(args.matched).open(encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        if not r.get("matched"):
            continue
        if args.level and r.get("best_level") != args.level:
            continue
        rows.append(r)
    print(f"[viz] matched rows in {Path(args.matched).name}: {len(rows):,}")
    if args.level:
        print(f"[viz] filtered to best_level={args.level}")

    # ── load top_sv_id + s_dino from gps_recovery ───────────────────
    gps = {}
    for line in Path(args.gps_recovery).open(encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        gps[(r["video"], r["frame_id"])] = r

    # ── load SV meta for crop file lookup ───────────────────────────
    sv_meta = {}
    for line in Path(args.sv_meta).open(encoding="utf-8"):
        if not line.strip():
            continue
        m = json.loads(line)
        sv_meta[m["id"]] = m

    # ── load heading_v2 decisions if present ────────────────────────
    heading_v2 = {}
    hv2_path = config.CITY_DIR / "a2" / "heading_v2.jsonl"
    if hv2_path.exists():
        for line in hv2_path.open(encoding="utf-8"):
            if not line.strip():
                continue
            d = json.loads(line)
            heading_v2[(d["video"], d["frame_id"])] = d
        print(f"[viz] heading_v2 rows: {len(heading_v2):,}")

    # ── load DINOv2 embeddings for on-the-fly 4-crop cosine ─────────
    cdir = config.CITY_DIR / "dinov2"
    sv_cache = np.load(cdir / "sv_v1.npz", allow_pickle=True)
    fr_cache = np.load(cdir / f"{args.frame_cache}.npz", allow_pickle=True)
    sv_embs = sv_cache["embs"]
    sv_ids = [str(s) for s in sv_cache["paths"]]
    sv_ids = [Path(s).stem for s in sv_ids]
    fr_embs = fr_cache["embs"]
    fr_paths = [Path(p) for p in fr_cache["paths"]]
    fr_idx = {(p.parent.name, p.stem): i for i, p in enumerate(fr_paths)}
    # group SV crops by pano_id for the 4-crop lookup
    pano_to_sv_idx = collections.defaultdict(list)
    for j, sid in enumerate(sv_ids):
        pid = sv_meta.get(sid, {}).get("pano_id", "")
        if pid:
            pano_to_sv_idx[pid].append(j)
    print(f"[viz] DINOv2 frame embeddings: {len(fr_embs)}, "
          f"SV embeddings: {len(sv_embs)}")

    # ── sample ─────────────────────────────────────────────────────
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    sample = rows[:args.n]
    print(f"[viz] rendering {len(sample)} rows")

    # ── render HTML ────────────────────────────────────────────────
    css = """
    body { font: 13px -apple-system, Segoe UI, Helvetica, sans-serif;
           background: #f5f5f5; margin: 16px; }
    h1 { font-size: 18px; margin: 0 0 12px; }
    .meta { color: #666; margin-bottom: 16px; }
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
    .info table { border-collapse: collapse; width: 100%; margin-top: 4px; }
    .info td { padding: 2px 6px; vertical-align: top; }
    .info td:first-child { color: #888; font-size: 11px; white-space: nowrap;
                            width: 80px; }
    .badge { display: inline-block; padding: 2px 8px; border-radius: 10px;
             font-size: 11px; font-weight: 600; }
    .good { color: #2a9d8f; font-weight: 600; }
    .ok   { color: #e9c46a; font-weight: 600; }
    .bad  { color: #c0392b; font-weight: 600; }
    .badge.attraction { background: #2a9d8f; color: #fff; }
    .badge.landmark { background: #e9c46a; color: #333; }
    .badge.poi { background: #ccc; color: #333; }
    .badge.exact { background: #2a9d8f; color: #fff; }
    .badge.substring { background: #f4a261; color: #333; }
    .badge.word_share { background: #e76f51; color: #fff; }
    .match-row { font-family: ui-monospace, Menlo, Consolas, monospace;
                 font-size: 11px; margin: 2px 0; }
    .name { font-weight: 600; }
    code { background: #eee; padding: 1px 4px; border-radius: 3px; }
    """

    html = ['<!DOCTYPE html><html><head><meta charset="utf-8">']
    html.append('<title>a2 — VLM-agreed matched frames (cos≥0.75)</title>')
    html.append(f'<style>{css}</style></head><body>')
    html.append(f'<h1>STEP 4 — matched frames from GPS_VLM_GEO.jsonl '
                f'(random sample of {len(sample)})</h1>')
    html.append('<div class="meta">QUERY frame (blue) on the left, '
                'top-1 DINOv2-matched Street View crop in the middle, '
                'GPS-side + VLM-side + matches on the right. '
                'Use this to verify that "matched" frames are actually at '
                'the right place (not DINOv2 lookalike false-positives).</div>')

    for r in sample:
        video, fid = r["video"], r["frame_id"]
        frame_path = config.FRAMES_DIR / video / f"{fid}.jpg"
        gps_row = gps.get((video, fid), {})
        top_sv = gps_row.get("top_sv_id", "")
        s_dino = gps_row.get("s_dino", 0.0)
        heading = gps_row.get("heading") or 0
        h_spread = gps_row.get("heading_spread") or 0
        h_gap = gps_row.get("heading_gap") or 0
        # The 4 compass crops at top-1's pano + their cosines
        crops = []
        if (video, fid) in fr_idx and top_sv:
            sims = sv_embs @ fr_embs[fr_idx[(video, fid)]]
            top_pano = sv_meta.get(top_sv, {}).get("pano_id", "")
            for j in pano_to_sv_idx.get(top_pano, []):
                h = sv_meta[sv_ids[j]].get("compass_angle", 0)
                crops.append((h, float(sims[j]), sv_ids[j]))
            crops.sort(key=lambda t: t[0])      # N→E→S→W
        best_cos = max((c for _, c, _ in crops), default=0.0)

        # GPS-side info
        ga = r["list_a_gps"].get("attractions", [])
        gl = r["list_a_gps"].get("landmarks", [])
        gp = r["list_a_gps"].get("pois", [])
        # VLM-side info
        va = r["list_b_vlm"].get("attractions", [])
        vl = r["list_b_vlm"].get("landmarks", [])
        vp = r["list_b_vlm"].get("pois", [])

        best = r.get("best_level") or ""
        best_badge = f'<span class="badge {best}">{best}</span>'

        # match rows
        match_html = []
        for m in r["matches"][:15]:
            mt = m["match_type"].split(":")[0]
            match_html.append(
                f'<div class="match-row">'
                f'<span class="badge {mt}">{mt}</span> '
                f'GPS=<span class="name">{m["gps_name"]}</span> '
                f'<span class="badge {m["gps_level"]}">{m["gps_level"]}</span>'
                f' &nbsp;⇄&nbsp; '
                f'VLM=<span class="name">{m["vlm_name"]}</span> '
                f'<span class="badge {m["vlm_level"]}">{m["vlm_level"]}</span>'
                f'</div>')
        if len(r["matches"]) > 15:
            match_html.append(
                f'<div class="match-row">(+{len(r["matches"])-15} more)</div>')

        # heading v2 info
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

        html.append('<div class="row">')
        # QUERY cell
        html.append(
            f'<div class="cell q">'
            f'<img src="{_file_url(frame_path)}" loading="lazy">'
            f'<div class="label"><b>QUERY</b><br>{video}/{fid}<br>'
            f'heading: <b>{h_v2_str}</b><br>'
            f'decision: {dec_str}<br>'
            f'best={best_badge}</div>'
            f'</div>')
        # The 4 compass crops at the matched pano — chosen one in red
        for h, cos, sid in crops:
            sv_path = config.STREETVIEW_DIR / "images" / f"{sid}.jpg"
            is_best = abs(cos - best_cos) < 1e-9
            cls = "good" if cos > 0.75 else ("ok" if cos >= 0.60 else "bad")
            style = (' style="outline: 3px solid #d62728; '
                     'outline-offset: -3px;"' if is_best else '')
            star = " ★" if is_best else ""
            html.append(
                f'<div class="cell">'
                f'<img src="{_file_url(sv_path)}" loading="lazy"{style}>'
                f'<div class="label">h <b>{int(h):03d}°</b>{star}<br>'
                f'<span class="{cls}">cos {cos:.3f}</span></div>'
                f'</div>')
        # pad to 4 SV cells if pano had fewer crops
        for _ in range(max(0, 4 - len(crops))):
            html.append('<div class="cell"><div class="label">—</div></div>')

        # info cell
        info = ['<div class="info"><table>']
        info.append(f'<tr><td>GPS attr</td><td>{", ".join(ga) or "—"}</td></tr>')
        info.append(f'<tr><td>GPS land</td><td>{", ".join(gl[:6]) or "—"}'
                    f'{" …" if len(gl)>6 else ""}</td></tr>')
        info.append(f'<tr><td>GPS pois</td><td>{", ".join(gp[:6]) or "—"}'
                    f'{" …" if len(gp)>6 else ""}</td></tr>')
        info.append(f'<tr><td>VLM attr</td><td>{", ".join(va) or "—"}</td></tr>')
        info.append(f'<tr><td>VLM land</td><td>{", ".join(vl[:6]) or "—"}'
                    f'{" …" if len(vl)>6 else ""}</td></tr>')
        info.append(f'<tr><td>VLM pois</td><td>{", ".join(vp[:6]) or "—"}'
                    f'{" …" if len(vp)>6 else ""}</td></tr>')
        info.append('</table>')
        info.append('<div style="margin-top:6px;"><b>matches:</b></div>')
        info.extend(match_html)
        info.append('</div>')
        html.append("".join(info))
        html.append('</div>')

    html.append('</body></html>')

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = Path.cwd() / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(html), encoding="utf-8")
    print(f"[viz] wrote {out_path}  ({len(sample)} rows)")
    print(f"      open: {_file_url(out_path)}")


if __name__ == "__main__":
    main()
