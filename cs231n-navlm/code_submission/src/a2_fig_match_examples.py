"""Generate docs/figures/fig4_match_success_failure.png — a 2x2 grid:
  col 1 = raw query frame
  col 2 = its DINOv2 top-1 StreetView crop (the "matched" location)
  row 1 = SUCCESS case (visually identical → correct mapping)
  row 2 = FAILURE case (visually similar but actually a different physical
          location → DINOv2 lookalike, 'noisy localization')

For the report's 'noisy localization' paragraph.
"""
import io
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config                                       # noqa: E402

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

GPS_RECOVERY = Path("data/cities/zurich/gps_recovery_full.jsonl")
FRAMES_ROOT = Path("data/cities/zurich/frames")
SV_ROOT = config.STREETVIEW_DIR / "images"

CASES = [
    # (row, video, frame, label, color, summary)
    (0, "hidden_streets", "frame_01894", "SUCCESS",
     "tab:green",
     "DINOv2 cosine 0.82 → matched StreetView crop is the SAME location "
     "(Münsterhof, in front of Fraumünster)."),
    (1, "bahnhofstrasse", "frame_01593", "FAILURE",
     "tab:red",
     "DINOv2 cosine 0.79 → matched StreetView crop is a DIFFERENT physical "
     "location with similar Zurich old-town facade (lookalike)."),
]

# Find top_sv_id per target frame
targets = {(v, f): {"label": l, "color": c, "summary": s}
            for _r, v, f, l, c, s in CASES}
for line in GPS_RECOVERY.open(encoding="utf-8"):
    r = json.loads(line)
    k = (r["video"], r["frame_id"])
    if k in targets:
        targets[k]["top_sv"] = r["top_sv_id"]
        targets[k]["cos"] = r["s_dino"]

# Query frames are 16:9 (≈1.78), SV crops are 1:1.
# Width-ratios match those aspect ratios so each image's axis is
# exactly the right shape and no inner-axis whitespace appears.
fig, axes = plt.subplots(2, 2, figsize=(11, 8.3), dpi=300,
                          gridspec_kw={"width_ratios": [1.78, 1.0]})

for row, (vid, fid), info in [(0, ("hidden_streets", "frame_01894"), targets[("hidden_streets", "frame_01894")]),
                                (1, ("bahnhofstrasse", "frame_01593"), targets[("bahnhofstrasse", "frame_01593")])]:
    label = info["label"]
    color = info["color"]
    summary = info["summary"]
    cos = info.get("cos", 0)
    sv_id = info.get("top_sv", "")

    # Left: raw query frame
    ax = axes[row, 0]
    frame_path = FRAMES_ROOT / vid / f"{fid}.jpg"
    ax.imshow(Image.open(frame_path).convert("RGB"))
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)
    ax.text(0.02, 0.97, label, transform=ax.transAxes,
            fontsize=22, fontweight="bold", color="white", va="top",
            bbox=dict(facecolor=color, edgecolor="none", pad=6))
    ax.text(0.98, 0.97, f"{vid}/{fid}", transform=ax.transAxes,
            fontsize=16, color="white", va="top", ha="right",
            bbox=dict(facecolor="black", edgecolor="none", pad=4, alpha=0.6))
    ax.set_xlabel("query frame (walking-tour video)", fontsize=20)
    if row == 0:
        ax.set_title("Raw query frame", fontsize=22)

    # Right: matched StreetView crop
    ax = axes[row, 1]
    sv_path = SV_ROOT / f"{sv_id}.jpg"
    if sv_path.exists():
        ax.imshow(Image.open(sv_path).convert("RGB"))
    else:
        ax.text(0.5, 0.5, f"(missing {sv_path})", ha="center", va="center",
                transform=ax.transAxes, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)
    ax.text(0.02, 0.97, f"cos={cos:.3f}", transform=ax.transAxes,
            fontsize=20, fontweight="bold", color="white", va="top",
            bbox=dict(facecolor="black", edgecolor="none", pad=5, alpha=0.6))
    # (SV ID removed — long opaque hash, not informative)
    ax.set_xlabel("DINOv2 top-1 match", fontsize=20)
    if row == 0:
        ax.set_title("Matched StreetView crop", fontsize=22)

# Row label on the LEFT margin
fig.text(0.025, 0.72, "SUCCESS",
         fontsize=24, fontweight="bold", color="tab:green",
         rotation=90, va="center")
fig.text(0.025, 0.28, "FAILURE",
         fontsize=24, fontweight="bold", color="tab:red",
         rotation=90, va="center")

plt.subplots_adjust(left=0.06, right=0.995, top=0.96, bottom=0.04,
                     wspace=0.0, hspace=0.18)
out_path = Path("docs/figures/fig4_match_success_failure.png")
out_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out_path, bbox_inches="tight")
plt.close()
print(f"saved {out_path}")
