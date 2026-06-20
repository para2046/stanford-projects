"""Score Modal eval outputs from `src/a2_eval_modal.py`.

Each per_sample.jsonl row carries a `model_response` and a `gt_verb`.
This script parses the response, computes the 4 metrics, and writes:

  - per_sample_scored.jsonl    — per-row scored copy
  - summary.json               — aggregated rates per condition
  - summary_table.txt          — printable table across conditions

Metrics:
  format_pass            : <thinking>+<answer> + valid first verb
  direction_pass         : first verb == gt_verb
  PASS                   : format_pass AND direction_pass
  heading_inference_acc  : only for *-derived conditions — |claimed - true|
                            < 22.5°

  python -m src.a2_score --run-dir eval_pull/<run_id>
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


VERBS = ("continue ahead", "turn left", "turn right", "turn around")


def parse_response(text):
    """Returns dict: format_pass, first_verb, answer_text, thinking_text,
    derived_heading."""
    out = {"format_pass": False, "first_verb": None,
           "answer_text": "", "thinking_text": "",
           "derived_heading": None}
    t_open = text.find("<thinking>")
    a_open = text.find("<answer>")
    if t_open >= 0:
        t_close = text.find("</thinking>", t_open)
        if t_close >= 0:
            out["thinking_text"] = text[t_open + 10:t_close].strip()
        elif a_open > t_open:
            out["thinking_text"] = text[t_open + 10:a_open].strip()
    if a_open >= 0:
        a_close = text.find("</answer>", a_open)
        if a_close >= 0:
            out["answer_text"] = text[a_open + 8:a_close].strip()
        else:
            out["answer_text"] = text[a_open + 8:].strip()
    # extract first verb (longest-match-first, case-insensitive)
    if out["answer_text"]:
        earliest = float("inf"); first = None
        for v in sorted(VERBS, key=len, reverse=True):
            m = re.search(rf"\b{re.escape(v)}\b",
                           out["answer_text"], re.I)
            if m and m.start() < earliest:
                first, earliest = v, m.start()
        out["first_verb"] = first
    # format_pass: both tags opened, valid verb
    out["format_pass"] = (
        t_open >= 0 and a_open >= 0 and out["first_verb"] is not None)
    # derived heading (regex "facing X°" in thinking)
    m = re.search(r"facing\s+(\d{1,3}(?:\.\d+)?)\s*°",
                   out["thinking_text"], re.I)
    if m:
        try:
            out["derived_heading"] = float(m.group(1))
        except ValueError:
            pass
    return out


def _circular_diff(a, b):
    return abs(((a - b + 180) % 360) - 180)


def score_row(row):
    """row = a per_sample.jsonl row; returns scored dict."""
    parsed = parse_response(row.get("model_response", ""))
    gt_verb = row.get("gt_verb")
    is_derived = row.get("condition", "").endswith("-heading-derived")
    direction_pass = (parsed["first_verb"] == gt_verb
                       if parsed["first_verb"] else False)
    PASS = parsed["format_pass"] and direction_pass

    # heading_inference_acc — only meaningful for derived
    heading_inference_pass = None
    if is_derived and parsed["derived_heading"] is not None:
        true_h = row.get("heading")
        if true_h is not None:
            heading_inference_pass = (
                _circular_diff(parsed["derived_heading"], true_h) < 22.5)

    return {
        **row,
        "first_verb": parsed["first_verb"],
        "format_pass": parsed["format_pass"],
        "direction_pass": direction_pass,
        "PASS": PASS,
        "derived_heading": parsed["derived_heading"],
        "heading_inference_pass": heading_inference_pass,
        "answer_text": parsed["answer_text"],
    }


def aggregate(scored):
    n = len(scored)
    if n == 0:
        return {}
    fmt = sum(1 for r in scored if r["format_pass"])
    dirp = sum(1 for r in scored if r["direction_pass"])
    passes = sum(1 for r in scored if r["PASS"])
    hi = [r for r in scored if r["heading_inference_pass"] is not None]
    hi_pass = sum(1 for r in hi if r["heading_inference_pass"])
    return {
        "n_samples": n,
        "format_accuracy": fmt / n,
        "direction_accuracy": dirp / n,
        "PASS_rate": passes / n,
        "heading_inference_n": len(hi),
        "heading_inference_acc": (hi_pass / max(1, len(hi))),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--run-dir", required=True,
                    help="local directory containing one or more "
                         "<condition>/per_sample.jsonl files "
                         "(after `modal volume get navlm-eval ...`)")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        sys.exit(f"run-dir not found: {run_dir}")

    summaries = {}
    for per_sample_path in sorted(run_dir.glob("*/per_sample.jsonl")):
        condition = per_sample_path.parent.name
        rows = [json.loads(l) for l in per_sample_path.open(encoding="utf-8")
                if l.strip()]
        if not rows:
            print(f"[score] {condition}: SKIPPED (empty per_sample.jsonl "
                  f"— eval probably still in progress)")
            continue
        scored = [score_row(r) for r in rows]
        scored_path = per_sample_path.parent / "per_sample_scored.jsonl"
        with scored_path.open("w", encoding="utf-8") as f:
            for r in scored:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        summary = aggregate(scored)
        summary["condition"] = condition
        summary_path = per_sample_path.parent / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2))
        summaries[condition] = summary
        print(f"[score] {condition}: n={summary['n_samples']}  "
              f"PASS={summary['PASS_rate']*100:.1f}%")

    # printable table across all conditions
    print()
    print("=" * 78)
    print(f"{'condition':<28s} {'n':>5s} {'fmt':>7s} {'dir':>7s} "
          f"{'PASS':>7s} {'h_inf':>7s} {'h_n':>5s}")
    print("-" * 78)
    for cond in sorted(summaries):
        s = summaries[cond]
        print(f"{cond:<28s} {s['n_samples']:>5d} "
              f"{s['format_accuracy']*100:>6.1f}% "
              f"{s['direction_accuracy']*100:>6.1f}% "
              f"{s['PASS_rate']*100:>6.1f}% "
              f"{s['heading_inference_acc']*100:>6.1f}% "
              f"{s['heading_inference_n']:>5d}")

    table_path = run_dir / "summary_table.txt"
    table_path.write_text("(see stdout above)")
    print(f"\n[score] per-condition summaries written to "
          f"{run_dir}/<condition>/summary.json")


if __name__ == "__main__":
    main()
