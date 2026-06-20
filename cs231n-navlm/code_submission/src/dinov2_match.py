"""DINOv2 match test — embed v2 video frames + the v1 Street View images
(178 panos x 4 headings = 712 jpgs already on disk), cosine top-k, emit
an HTML grid for visual inspection.

This is the diagnostic that motivated the navlm_v2 redesign — "DINOv2
hard-codes two images to be alike". We want to see, with our actual
walking-tour frames and the already-bought SV images, how well DINOv2
matches before spending more on a new SV crawl.

    python -m src.dinov2_match                   # default: ~80 frames
    python -m src.dinov2_match --every-n 100     # denser sample
    python -m src.dinov2_match --limit 20 -k 5   # tiny trial, more matches
    python -m src.dinov2_match --sv "<dir>"      # different SV directory
"""

import argparse
import sys
from pathlib import Path
from urllib.parse import quote

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config                                   # noqa: E402
from src.gps_recovery import cosine_topk        # noqa: E402

# Default Street View directory — the v2 canonical local path. The v1
# 712 images (178 panos x 4 headings) have been copied here from Drive
# to avoid the cloud-fetch latency that hung an earlier run.
SV_DIR_DEFAULT = config.STREETVIEW_DIR / "images"


def load_embedder(model_name=config.DINOV2_MODEL):
    """Load DINOv2 and its processor. Returns (model, processor, device).
    Stage prints flush after each step so hangs are visible."""
    print("[dinov2] importing torch...", flush=True)
    import torch
    print("[dinov2] torch ready, importing transformers...", flush=True)
    from transformers import AutoModel, AutoImageProcessor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[dinov2] loading processor ({model_name})...", flush=True)
    proc = AutoImageProcessor.from_pretrained(model_name)
    print(f"[dinov2] loading model -> {device}...", flush=True)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    print(f"[dinov2] model ready on {device}", flush=True)
    return model, proc, device


def embed_images(paths, model, proc, device, batch_size=16):
    """Embed a list of image paths -> (N, D) float32, L2-normalised.
    Uses the DINOv2 CLS token (last_hidden_state[:, 0, :])."""
    import torch
    from PIL import Image
    feats = []
    for i in tqdm(range(0, len(paths), batch_size),
                  desc="[embed]", unit="batch"):
        batch = [Image.open(p).convert("RGB")
                 for p in paths[i:i + batch_size]]
        inputs = proc(images=batch, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**inputs)
        emb = out.last_hidden_state[:, 0, :].cpu().numpy()
        feats.append(emb)
    feats = np.vstack(feats).astype(np.float32)
    feats /= np.linalg.norm(feats, axis=1, keepdims=True) + 1e-9
    return feats


def discover_sv_images(sv_dir):
    """All .jpg in sv_dir, sorted by name. Pure."""
    return sorted(Path(sv_dir).glob("*.jpg"))


def discover_frames(every_n):
    """Every Nth extracted frame across all videos, sorted within video."""
    out = []
    if not config.FRAMES_DIR.exists():
        return out
    for vdir in sorted(config.FRAMES_DIR.iterdir()):
        if not vdir.is_dir() or vdir.name.endswith("_dense"):
            continue
        out.extend(sorted(vdir.glob("frame_*.jpg"))[::every_n])
    return out


def cached_embed(name, paths, model, proc, device):
    """Embed if the cache is missing or stale; save to
    `data/cities/zurich/dinov2/{name}.npz`. Returns embeddings."""
    cache_dir = config.CITY_DIR / "dinov2"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"{name}.npz"
    path_strs = np.array([str(p) for p in paths])
    if cache.exists():
        d = np.load(cache, allow_pickle=True)
        if (len(d["paths"]) == len(paths) and
                (d["paths"] == path_strs).all()):
            print(f"[cache] reused {cache.name} ({len(paths)} images)",
                  flush=True)
            return d["embs"]
    print(f"[embed] {name}: {len(paths)} images", flush=True)
    embs = embed_images(paths, model, proc, device)
    np.savez(cache, embs=embs, paths=path_strs)
    print(f"[cache] saved {cache.name}", flush=True)
    return embs


def _file_url(p):
    """file:/// URL for a local path, with %20 for spaces (Google Drive)."""
    return "file:///" + quote(str(Path(p).resolve()).replace("\\", "/"))


def build_viz(frames, sv_paths, frame_embs, sv_embs, k, min_sim, out_path):
    """HTML grid sorted by top-1 cosine descending, with a divider
    between MATCHED (top-1 >= min_sim) and NO MATCH frames."""
    # precompute top-K once per frame, sort by top-1 desc
    per_frame = []
    for i, frame in enumerate(frames):
        idx, sims = cosine_topk(frame_embs[i], sv_embs, k=k)
        per_frame.append((float(sims[0]), frame, idx, sims))
    per_frame.sort(key=lambda r: r[0], reverse=True)
    n_matched = sum(1 for r in per_frame if r[0] >= min_sim)

    def row_html(top1, frame, idx, sims):
        cells = [
            f'<div class="cell q"><img src="{_file_url(frame)}" loading="lazy">'
            f'<div class="label"><b>QUERY</b><br>'
            f'{Path(frame).parent.name}/{Path(frame).name}<br>'
            f'top-1 cos {top1:.3f}</div></div>'
        ]
        for j, s in zip(idx, sims):
            cls = "good" if s > 0.75 else ("ok" if s >= min_sim else "bad")
            sv = Path(sv_paths[int(j)])
            cells.append(
                f'<div class="cell"><img src="{_file_url(sv)}" loading="lazy">'
                f'<div class="label">{sv.name[:30]}<br>'
                f'<span class="{cls}">cos {float(s):.3f}</span></div></div>')
        return '<div class="row">' + ''.join(cells) + '</div>'

    matched_rows = [row_html(*r) for r in per_frame if r[0] >= min_sim]
    unmatched_rows = [row_html(*r) for r in per_frame if r[0] < min_sim]

    divider_matched = (
        f'<div class="divider good">'
        f'MATCHED &mdash; top-1 cosine &ge; {min_sim} '
        f'({len(matched_rows)} / {len(per_frame)} frames, '
        f'{100*len(matched_rows)/max(1,len(per_frame)):.0f}%)</div>')
    divider_unmatched = (
        f'<div class="divider bad">'
        f'NO MATCH &mdash; top-1 cosine &lt; {min_sim} '
        f'({len(unmatched_rows)} / {len(per_frame)} frames, '
        f'{100*len(unmatched_rows)/max(1,len(per_frame)):.0f}%)</div>')

    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>NavLM v2 - DINOv2 vs v1 Street View</title>
<style>
body {{ font-family: Arial, sans-serif; background: #f5f5f5; margin: 16px; }}
h1 {{ color: #1a1a1a; }}
.stats {{ background: #e9edf4; padding: 10px 14px; border-radius: 6px;
         margin-bottom: 16px; }}
.divider {{ padding: 10px 14px; border-radius: 6px; color: white;
           font-weight: bold; margin: 18px 0 10px 0; }}
.divider.good {{ background: #2d7d2d; }}
.divider.bad  {{ background: #cc3333; }}
.row {{ display: flex; gap: 8px; margin-bottom: 10px; background: white;
        padding: 8px; border-radius: 6px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
.cell {{ text-align: center; }}
.cell img {{ width: 240px; height: auto; border-radius: 4px; }}
.cell.q img {{ border: 3px solid #1f3a68; }}
.cell:not(.q) img {{ border: 2px solid #ccc; }}
.label {{ font-size: 11px; color: #555; margin-top: 4px; }}
.good {{ color: #2d7d2d; font-weight: bold; }}
.ok   {{ color: #b58900; }}
.bad  {{ color: #cc3333; }}
</style></head><body>
<h1>DINOv2 match: v2 video frames &harr; v1 Street View ({len(sv_paths)} images)</h1>
<div class="stats">
<b>Frames tested:</b> {len(frames)} &nbsp;|&nbsp;
<b>SV reference:</b> {len(sv_paths)} images (178 panos &times; 4 headings) &nbsp;|&nbsp;
<b>Top-K shown:</b> {k} &nbsp;|&nbsp;
<b>Match threshold:</b> cos &ge; {min_sim} &nbsp;|&nbsp;
<b>Match rate:</b>
<span class="good">{100*n_matched/max(1,len(per_frame)):.0f}%</span>
({n_matched}/{len(per_frame)})<br>
Rows sorted by top-1 cosine desc. Cell colour:
<span class="good">&gt; 0.75 strong</span> &middot;
<span class="ok">&ge; threshold</span> &middot;
<span class="bad">&lt; threshold</span>
</div>
{divider_matched}
{''.join(matched_rows)}
{divider_unmatched}
{''.join(unmatched_rows)}
</body></html>"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print(f"wrote {out_path}", flush=True)


def main():
    ap = argparse.ArgumentParser(
        description="DINOv2 match — v2 frames vs v1 Street View")
    ap.add_argument("--sv", type=str, default=str(SV_DIR_DEFAULT),
                    help="Street View images dir (default: local v2 path)")
    ap.add_argument("--every-n", type=int, default=300,
                    help="sample every Nth extracted frame")
    ap.add_argument("--limit", type=int, default=0,
                    help="cap test frames (0 = no cap)")
    ap.add_argument("-k", type=int, default=3,
                    help="top-K matches per frame")
    ap.add_argument("--min-sim", type=float, default=0.60,
                    help="cosine threshold for a 'real' match (default 0.60)")
    args = ap.parse_args()

    sv_paths = discover_sv_images(args.sv)
    if not sv_paths:
        sys.exit(f"no SV images at {args.sv}")
    frames = discover_frames(args.every_n)
    if args.limit:
        frames = frames[:args.limit]
    if not frames:
        sys.exit(f"no frames at {config.FRAMES_DIR} — run extract_frames first")

    print(f"SV reference: {len(sv_paths)} images from {args.sv}", flush=True)
    print(f"test frames:  {len(frames)} (every {args.every_n}-th, "
          f"capped {args.limit})", flush=True)

    model, proc, device = load_embedder()
    print(f"DINOv2 on {device}", flush=True)

    sv_embs = cached_embed("sv_v1", sv_paths, model, proc, device)
    frame_embs = cached_embed(
        f"frames_n{args.every_n}_l{args.limit}",
        frames, model, proc, device)

    # top-1 cosine distribution + match rate at the threshold
    top1 = np.array([float(cosine_topk(frame_embs[i], sv_embs, k=1)[1][0])
                     for i in range(len(frames))])
    print(f"\ntop-1 cosine: mean={top1.mean():.3f}  "
          f"median={float(np.median(top1)):.3f}  "
          f"min={top1.min():.3f}  max={top1.max():.3f}", flush=True)
    bins = [(0.85, ">0.85 very strong"), (0.75, ">0.75 strong"),
            (args.min_sim, f">={args.min_sim} matched (threshold)"),
            (0.50, ">0.50 weak"), (0.0, ">=0.0 all")]
    for thr, lab in bins:
        n = int((top1 >= thr).sum())
        print(f"  {lab:35s} {n:4d}/{len(top1)}  "
              f"({100 * n / len(top1):.0f}%)", flush=True)

    out = config.VIZ_DIR / "dinov2_match_test.html"
    build_viz(frames, sv_paths, frame_embs, sv_embs,
              args.k, args.min_sim, out)


if __name__ == "__main__":
    main()
