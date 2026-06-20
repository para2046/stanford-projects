"""V-stripped figures: regenerate fig1/2/3 + extra fig{3b,3c} from nov eval pulls.

Inputs : eval_pull/nov_<run_id>/<dir>/per_sample.jsonl   (one per condition)
Outputs: docs/figures/fig{1,2,3}_*.png        (overwrites the previous-run figures)
         docs/figures/fig3b_derived_err_by_distance.png
         docs/figures/fig3c_derived_err_by_attraction.png

Also writes:
  docs/figures/_nov_stats.txt   — text dump of all 21 conditions + extra tables

  python -m src.a2_figures_nov  --run-dir eval_pull/nov_20260604_223758
"""
from __future__ import annotations

import argparse
import collections
import io
import json
import os
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.a2_score import score_row              # noqa: E402

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

HEAD_RE = re.compile(r"facing\s+(\d{1,3}(?:\.\d+)?)\s*°", re.I)
STEPS = {"given": 321, "derived": 266, "implicit": 268}


def load_nov_summary(run_dir: Path) -> dict:
    """Returns {(typ, variant, rank, epoch): {n, PASS, rows}} dedup-by-best-n."""
    cand = {}
    for d in sorted(run_dir.iterdir()):
        if not d.is_dir():
            continue
        p = d / "per_sample.jsonl"
        if not p.exists() or p.stat().st_size == 0:
            continue
        rows_raw = [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]
        if not rows_raw:
            continue
        rows = [score_row(r) for r in rows_raw]
        n = len(rows)
        first = rows_raw[0]
        adapter = first.get("adapter") or ""
        if not adapter:
            # zero-shot
            variant = first.get("variant", "")
            ident = ("zs", variant, 0, 0)
        else:
            m_rk = re.search(r"lora_a2_(\w+?)_r(\d+)_e\d+", adapter)
            m_ep = re.search(r"checkpoint-(\d+)", adapter)
            if not (m_rk and m_ep):
                continue
            variant = m_rk.group(1)
            rk = int(m_rk.group(2))
            ck = int(m_ep.group(1))
            ep = ck // STEPS[variant]
            ident = ("trained", variant, rk, ep)
        entry = {
            "n": n,
            "PASS": sum(1 for r in rows if r.get("PASS")) / n,
            "dir":  sum(1 for r in rows if r.get("direction_pass")) / n,
            "fmt":  sum(1 for r in rows if r.get("format_pass")) / n,
            "rows": rows,                                  # already scored
        }
        if ident not in cand or n > cand[ident]["n"]:
            cand[ident] = entry
    return cand


# ─────────────────────── Figure 1: rank-saturation ─────────────────────────
def fig1_rank_saturation(summary, out_path: Path):
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
    variants = ["given", "derived", "implicit"]
    ranks = [4, 8, 16]
    colors = {"given": "#1f77b4", "derived": "#ff7f0e", "implicit": "#2ca02c"}
    epoch_styles = {3: {"linestyle": "--", "marker": "o", "label_suffix": " (e3)"},
                     5: {"linestyle": "-",  "marker": "s", "label_suffix": " (e5)"}}

    for v in variants:
        for e, sty in epoch_styles.items():
            ys = []
            for r in ranks:
                key = ("trained", v, r, e)
                ys.append(summary[key]["PASS"] * 100 if key in summary else np.nan)
            ax.plot(ranks, ys, color=colors[v],
                    label=f"{v}{sty['label_suffix']}",
                    linestyle=sty["linestyle"], marker=sty["marker"],
                    markersize=7, linewidth=1.8)

    for v in variants:
        zs = summary.get(("zs", v, 0, 0), {}).get("PASS")
        if zs is not None:
            ax.axhline(zs * 100, color=colors[v], linestyle=":",
                        linewidth=1, alpha=0.5)
            ax.text(16.3, zs * 100, f"zs-{v}", color=colors[v],
                     fontsize=8, va="center", ha="left")

    ax.set_xlabel("LoRA rank", fontsize=11)
    ax.set_ylabel("PASS rate (%)", fontsize=11)
    ax.set_title("Rank-saturation: LoRA rank vs PASS, per variant × epoch count",
                  fontsize=12, pad=10)
    ax.set_xticks(ranks)
    ax.set_xticklabels(["r=4", "r=8", "r=16"])
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
               frameon=False, fontsize=9)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout(); plt.savefig(out_path, bbox_inches="tight"); plt.close()
    print(f"saved {out_path}")


# ───────────── Figure 2: zero-shot vs best-trained bar chart ───────────────
def fig2_zs_vs_trained(summary, out_path: Path):
    fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=300)
    variants = ["given", "derived", "implicit"]
    zs_vals, tr_vals, tr_labels = [], [], []
    for v in variants:
        zs = summary.get(("zs", v, 0, 0), {}).get("PASS", 0) * 100
        best, best_key = -1, None
        for r in [4, 8, 16]:
            for e in [3, 5]:
                key = ("trained", v, r, e)
                if key in summary:
                    p = summary[key]["PASS"] * 100
                    if p > best:
                        best, best_key = p, key
        zs_vals.append(zs); tr_vals.append(best)
        if best_key:
            tr_labels.append(f"r={best_key[2]} e{best_key[3]}")
        else:
            tr_labels.append("?")

    x = np.arange(len(variants)); w = 0.35
    bars_zs = ax.bar(x - w/2, zs_vals, w, label="zero-shot",
                      color="#bbbbbb", edgecolor="black", linewidth=0.6)
    bars_tr = ax.bar(x + w/2, tr_vals, w, label="best LoRA",
                      color=["#1f77b4", "#ff7f0e", "#2ca02c"],
                      edgecolor="black", linewidth=0.6)
    for b, v in zip(bars_zs, zs_vals):
        ax.text(b.get_x() + b.get_width()/2, v + 1, f"{v:.1f}%",
                 ha="center", fontsize=9)
    for b, v, lbl in zip(bars_tr, tr_vals, tr_labels):
        ax.text(b.get_x() + b.get_width()/2, v + 1, f"{v:.1f}%",
                 ha="center", fontsize=9, fontweight="bold")
        ax.text(b.get_x() + b.get_width()/2, v / 2, lbl,
                 ha="center", fontsize=8, color="white", fontweight="bold")
    for i, (zs, tr) in enumerate(zip(zs_vals, tr_vals)):
        ax.annotate("", xy=(i + w/2, tr), xytext=(i - w/2, zs),
                     arrowprops=dict(arrowstyle="->", color="red", alpha=0.4, lw=1))
        ax.text(i, max(zs, tr) + 6, f"+{tr - zs:.1f}pp",
                 ha="center", fontsize=10, color="red", fontweight="bold")

    ax.set_ylabel("PASS rate (%)", fontsize=11)
    ax.set_title("Zero-shot vs best LoRA-trained: Qwen 2.5 VL 7B on Zurich navigation",
                  fontsize=11, pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(["heading-given", "heading-derived", "heading-implicit"],
                        fontsize=10)
    ax.set_ylim(0, 120)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.grid(axis="y", alpha=0.3, linestyle="-", linewidth=0.5)
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout(); plt.savefig(out_path, bbox_inches="tight"); plt.close()
    print(f"saved {out_path}")


def collect_heading_errors(rows):
    out = []
    for r in rows:
        resp = r.get("model_response", "")
        t1 = resp.find("<thinking>"); t2 = resp.find("</thinking>", t1)
        if t1 < 0: continue
        thinking = resp[t1+10: t2 if t2 > t1 else None]
        m = HEAD_RE.search(thinking)
        if not m: continue
        try:
            p = float(m.group(1))
        except ValueError:
            continue
        g = r.get("heading")
        if g is None: continue
        e = abs(((p - g + 180) % 360) - 180)
        out.append((g, p, e, r))
    return out


# ──────────── Figure 3: derived heading-inference scatter ──────────────────
def fig3_heading_scatter(summary, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), dpi=300, sharey=True)
    panels = [
        (("zs", "derived", 0, 0),         "Zero-shot Qwen"),
        (("trained", "derived", 16, 5),   "Trained derived (r=16, e=5)"),
    ]
    for ax, (key, title) in zip(axes, panels):
        rows = summary[key]["rows"]
        data = collect_heading_errors(rows)
        if not data:
            continue
        gt = np.array([d[0] for d in data])
        pred = np.array([d[1] for d in data])
        err = np.array([d[2] for d in data])
        n = len(gt); n_total = summary[key]["n"]
        ax.plot([0, 360], [0, 360], "k--", alpha=0.4, linewidth=1)
        for delta in [22.5, -22.5]:
            ax.plot([0, 360], [delta, 360+delta], "k:", alpha=0.2, linewidth=0.8)
        colors = np.where(err < 22.5, "tab:green",
                           np.where(err < 90, "tab:orange", "tab:red"))
        ax.scatter(gt, pred, c=colors, s=18, alpha=0.65, edgecolors="none")
        within_22 = (err < 22.5).sum() / n * 100
        mean_err = err.mean()
        emit_pct = n / n_total * 100
        ax.text(0.05, 0.95,
                 f"n={n}/{n_total} emit ({emit_pct:.0f}%)\n"
                 f"within 22.5°: {within_22:.1f}%\n"
                 f"mean |err|: {mean_err:.1f}°",
                 transform=ax.transAxes, fontsize=9, va="top",
                 bbox=dict(facecolor="white", edgecolor="black",
                            boxstyle="round,pad=0.3", alpha=0.85))
        ax.set_xlim(0, 360); ax.set_ylim(0, 360)
        ax.set_xlabel("GT heading (°)", fontsize=10)
        if ax is axes[0]:
            ax.set_ylabel("Predicted heading (°)", fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.set_xticks([0, 90, 180, 270, 360])
        ax.set_yticks([0, 90, 180, 270, 360])
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    handles = [
        plt.Line2D([0],[0], marker="o", color="w",
                    markerfacecolor="tab:green", markersize=8, label="< 22.5° (verb-correct)"),
        plt.Line2D([0],[0], marker="o", color="w",
                    markerfacecolor="tab:orange", markersize=8, label="22.5-90° (close-ish)"),
        plt.Line2D([0],[0], marker="o", color="w",
                    markerfacecolor="tab:red", markersize=8, label="> 90° (wildly off)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3,
                bbox_to_anchor=(0.5, -0.02), frameon=False, fontsize=9)
    fig.suptitle("Heading inference: derived model's 'facing X°' vs GT",
                  fontsize=12)
    plt.tight_layout(rect=[0, 0.04, 1, 0.95])
    plt.savefig(out_path, bbox_inches="tight"); plt.close()
    print(f"saved {out_path}")


# ─────────── Figure 3b: derived error rate vs route distance ───────────
def fig3b_err_by_distance(summary, out_path: Path):
    key = ("trained", "derived", 16, 3)             # best derived
    rows = summary[key]["rows"]
    bins = [(0, 100), (100, 200), (200, 400), (400, 800), (800, 1e9)]
    bin_lbls = ["0-100 m", "100-200 m", "200-400 m", "400-800 m", "800 m+"]
    counts = [0] * len(bins); passes = [0] * len(bins)
    for r in rows:
        d = r.get("route_distance_m") or r.get("distance_m") or None
        if d is None:
            continue
        for i, (lo, hi) in enumerate(bins):
            if lo <= d < hi:
                counts[i] += 1
                if r.get("PASS"):
                    passes[i] += 1
                break

    pcts = [p / c * 100 if c else 0 for p, c in zip(passes, counts)]
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=300)
    x = np.arange(len(bins))
    bars = ax.bar(x, pcts, color="#ff7f0e", edgecolor="black", linewidth=0.6)
    for b, pct, c in zip(bars, pcts, counts):
        ax.text(b.get_x() + b.get_width()/2, pct + 1.5,
                 f"{pct:.1f}%\n(n={c})", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(bin_lbls, fontsize=10)
    ax.set_ylabel("PASS rate (%)", fontsize=11)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_title("Derived (r=16, e=3) PASS rate by walking-route distance",
                  fontsize=11)
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout(); plt.savefig(out_path, bbox_inches="tight"); plt.close()
    print(f"saved {out_path}")
    return list(zip(bin_lbls, counts, passes, pcts))


# ────────── Figure 3c: derived error rate by destination attraction ──────
def fig3c_err_by_attraction(summary, out_path: Path):
    key = ("trained", "derived", 16, 3)
    rows = summary[key]["rows"]
    by_dest = collections.defaultdict(lambda: [0, 0])    # [n, passes]
    for r in rows:
        dest = r.get("destination") or "(unknown)"
        by_dest[dest][0] += 1
        if r.get("PASS"):
            by_dest[dest][1] += 1
    # keep top destinations by sample count
    items = sorted(by_dest.items(), key=lambda x: -x[1][0])
    items = items[:12]                                  # cap for readability
    labels = [d for d, _ in items]
    counts = [v[0] for _, v in items]
    pcts = [v[1] / v[0] * 100 if v[0] else 0 for _, v in items]

    fig, ax = plt.subplots(figsize=(8.5, 4.5), dpi=300)
    x = np.arange(len(labels))
    bars = ax.bar(x, pcts, color="#ff7f0e", edgecolor="black", linewidth=0.6)
    for b, pct, c in zip(bars, pcts, counts):
        ax.text(b.get_x() + b.get_width()/2, pct + 1.5,
                 f"{pct:.0f}%\n(n={c})", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
    ax.set_ylim(0, 110)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_ylabel("PASS rate (%)", fontsize=11)
    ax.set_title("Derived (r=16, e=3) PASS rate by destination attraction",
                  fontsize=11)
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout(); plt.savefig(out_path, bbox_inches="tight"); plt.close()
    print(f"saved {out_path}")
    return list(zip(labels, counts, pcts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="eval_pull/nov_20260604_223758")
    ap.add_argument("--out-dir", default="docs/figures")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    summary = load_nov_summary(run_dir)
    print(f"loaded {len(summary)} conditions from {run_dir}")

    # ─── enrich derived rows with route_distance_m + destination if missing
    # The per_sample.jsonl from Modal eval might already carry these; if
    # not, join with the test SFT file.
    sft_dir = Path("data/sft")
    for variant in ["derived"]:
        sft_test = sft_dir / f"a2_{variant}_test_nov.jsonl"
        if not sft_test.exists():
            continue
        idx = {}
        for line in sft_test.open(encoding="utf-8"):
            r = json.loads(line)
            idx[(r["video"], r["frame_id"])] = r
        for k, e in summary.items():
            typ, v, rk, ep = k
            if v != variant: continue
            for r in e["rows"]:
                src = idx.get((r["video"], r["frame_id"]))
                if not src: continue
                # destination + distance enrichment (distance from route file)
                if "destination" not in r:
                    r["destination"] = src.get("destination")
    # distance enrichment from routes.jsonl (use best-distance multi-target route per frame)
    routes_path = Path("data/cities/zurich/a2/routes.jsonl")
    if routes_path.exists():
        frame_dist = {}
        for line in routes_path.open(encoding="utf-8"):
            r = json.loads(line)
            k = (r["video"], r["frame_id"])
            d = r.get("route_distance_m")
            if d is None: continue
            if k not in frame_dist or d > frame_dist[k]:
                frame_dist[k] = d
        for k, e in summary.items():
            typ, v, rk, ep = k
            if v != "derived": continue
            for r in e["rows"]:
                d = frame_dist.get((r["video"], r["frame_id"]))
                if d is not None:
                    r["route_distance_m"] = d

    fig1_rank_saturation(summary, out_dir / "fig1_rank_saturation.png")
    fig2_zs_vs_trained(summary,   out_dir / "fig2_zs_vs_trained.png")
    fig3_heading_scatter(summary, out_dir / "fig3_heading_scatter.png")
    bin_stats = fig3b_err_by_distance(summary,
                                       out_dir / "fig3b_derived_err_by_distance.png")
    att_stats = fig3c_err_by_attraction(summary,
                                         out_dir / "fig3c_derived_err_by_attraction.png")

    # ─── write a stats dump used for the manual patch ──────────────────
    dump_path = out_dir / "_nov_stats.txt"
    with dump_path.open("w", encoding="utf-8") as f:
        f.write("=== V-stripped (_nov) eval — 21-condition PASS table ===\n\n")
        f.write(f"{'condition':<22s} {'n':>5s} {'PASS':>7s} {'dir':>7s} {'fmt':>7s}\n")
        order = [("zs",      v, 0, 0) for v in ["given","derived","implicit"]]
        for v in ["given","derived","implicit"]:
            for rk in [4, 8, 16]:
                for ep in [3, 5]:
                    order.append(("trained", v, rk, ep))
        for k in order:
            e = summary.get(k)
            typ, v, rk, ep = k
            lbl = f"zs-{v}" if typ == "zs" else f"{v}-r{rk}-e{ep}"
            if e:
                f.write(f"{lbl:<22s} {e['n']:>5d} "
                          f"{e['PASS']*100:>6.1f}% {e['dir']*100:>6.1f}% "
                          f"{e['fmt']*100:>6.1f}%\n")
            else:
                f.write(f"{lbl:<22s}  MISSING\n")

        # heading-error summary (derived only)
        f.write("\n=== Heading inference (derived) — |err| (°) ===\n")
        for key, title in [(("zs","derived",0,0), "zs-derived"),
                            (("trained","derived",16,5), "derived-r16-e5"),
                            (("trained","derived",16,3), "derived-r16-e3")]:
            if key not in summary: continue
            data = collect_heading_errors(summary[key]["rows"])
            if not data: continue
            errs = np.array([d[2] for d in data])
            n = len(errs); ntot = summary[key]["n"]
            f.write(f"{title:<18s}  emit={n}/{ntot}({n/ntot*100:.0f}%)  "
                      f"mean={errs.mean():.1f}°  median={np.median(errs):.1f}°  "
                      f"within22.5={((errs<22.5).sum()/n*100):.1f}%  "
                      f"within45={((errs<45).sum()/n*100):.1f}%\n")

        f.write("\n=== Derived (r16,e3) PASS by route distance ===\n")
        f.write(f"{'bin':<12s} {'n':>5s} {'pass':>5s}  PASS%\n")
        for lbl, n, p, pct in bin_stats:
            f.write(f"{lbl:<12s} {n:>5d} {p:>5d}  {pct:>5.1f}%\n")

        f.write("\n=== Derived (r16,e3) PASS by destination ===\n")
        f.write(f"{'destination':<24s} {'n':>5s}  PASS%\n")
        for lbl, n, pct in att_stats:
            f.write(f"{lbl:<24s} {n:>5d}  {pct:>5.1f}%\n")
    print(f"saved {dump_path}")


if __name__ == "__main__":
    main()
