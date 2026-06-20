"""Convert annotations_a2_{variant}.jsonl into Qwen2.5-VL chat-template
format that `src/a2_train_modal.py` consumes.

Each variant gets its own INDEPENDENT random 80/10/10 split — there is
NO cross-variant key alignment. Cross-condition comparison in §20 is
done on rates (PASS rate, format rate, etc.), not per-row deltas;
within-variant `zs-X` vs `trained-X` is paired automatically because
both load the same `a2_X_test.jsonl`.

Input:  data/cities/zurich/a2/annotations_a2_{variant}.jsonl
Output: data/sft/a2_{variant}_{split}.jsonl   (split ∈ {train, val, test})

Per-row output shape:
  {
    "image_rel": "<video>/<frame_id>.jpg",
    "messages": [
      {"role": "system", "content": [{"type":"text", "text": <SYSTEM_PROMPT>}]},
      {"role": "user",   "content": [
          {"type": "image"},
          {"type": "text", "text": <student_prompt>}
      ]},
      {"role": "assistant", "content": [
          {"type": "text", "text": <teacher_response>}
      ]}
    ],
    # diagnostic carry-over (not consumed by trainer):
    "video": ..., "frame_id": ..., "destination": ...,
    "gt_verb": ..., "first_verb": ..., "heading": ...
  }

Splits:
  - train/val/test = 80/10/10 by random shuffle (seed 42 by default)
  - only rows with format_pass==True are kept (skip teacher-malformed)
  - optional --only-pass keeps only direction_pass==True rows
    (cleaner SFT labels at ~0.7× the cohort size)
  - optional --holdout-video puts one entire video into TEST (rare; the
    default random split is the recommended setup)

  python -m src.a2_to_sft --variant given
  python -m src.a2_to_sft --variant derived --only-pass
  python -m src.a2_to_sft --variant implicit
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config                                       # noqa: E402
from src.a2_annotate import system_prompt          # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--variant", choices=["given", "derived", "implicit"],
                    required=True)
    ap.add_argument("--input", default=None,
                    help="annotations_a2_{variant}.jsonl path "
                         "(default: data/cities/zurich/a2/...)")
    ap.add_argument("--out-dir", default="data/sft",
                    help="where to write the split files")
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--test-frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--strip-visible", action="store_true",
                    help="strip the 'Visible landmarks at this spot: ...' "
                         "block from the student prompt. The teacher still "
                         "saw V at annotation time (so its answers may name "
                         "landmarks), but the student is trained / evaluated "
                         "WITHOUT V — forcing it to recognise landmarks "
                         "visually from the image itself.")
    ap.add_argument("--suffix", default="",
                    help="filename suffix for the output splits "
                         "(e.g. '_nov' produces a2_<v>_train_nov.jsonl etc.) "
                         "so a new rerun does not overwrite earlier files.")
    ap.add_argument("--only-pass", action="store_true",
                    help="keep only rows where direction_pass==True "
                         "(cleaner SFT but smaller; default: keep all "
                         "format_pass rows regardless of direction)")
    ap.add_argument("--holdout-video", default="",
                    help="if set, that video goes entirely to TEST; "
                         "the rest is randomly split into train/val")
    args = ap.parse_args()

    in_path = (Path(args.input) if args.input
               else config.CITY_DIR / "a2"
                    / f"annotations_a2_{args.variant}.jsonl")
    if not in_path.exists():
        sys.exit(f"input not found: {in_path}")

    rows = [json.loads(l) for l in in_path.open(encoding="utf-8")
            if l.strip()]
    print(f"[to_sft] loaded {len(rows):,} from {in_path.name}")

    # filter format failures (malformed teacher response)
    rows = [r for r in rows if r.get("format_pass")]
    print(f"[to_sft] after format_pass filter: {len(rows):,}")
    if args.only_pass:
        rows = [r for r in rows if r.get("direction_pass")]
        print(f"[to_sft] after direction_pass filter (--only-pass): "
              f"{len(rows):,}")

    sys_prompt_text = system_prompt(args.variant,
                                     strip_visible=args.strip_visible)

    # Pattern stripped when --strip-visible:
    #   "Visible landmarks at this spot:\n  Limmatquai, Münsterbrücke\n\n"
    #   "Visible landmarks at this spot:\n  (no notable landmarks listed)\n\n"
    # Compile once.
    import re as _re
    _vis_re = _re.compile(
        r"Visible landmarks at this spot:\n\s+[^\n]+\n\n", _re.MULTILINE)

    def _strip_visible(prompt):
        new, n = _vis_re.subn("", prompt)
        return new

    def _to_qwen_row(r):
        student_prompt = r["student_prompt"]
        if args.strip_visible:
            student_prompt = _strip_visible(student_prompt)
        return {
            "image_rel": f"{r['video']}/{r['frame_id']}.jpg",
            "messages": [
                {"role": "system",
                 "content": [{"type": "text", "text": sys_prompt_text}]},
                {"role": "user",
                 "content": [
                     {"type": "image"},
                     {"type": "text", "text": student_prompt},
                 ]},
                {"role": "assistant",
                 "content": [{"type": "text", "text": r["response"]}]},
            ],
            # diagnostic carry-over
            "video": r["video"], "frame_id": r["frame_id"],
            "destination": r["destination"],
            "destination_zh": r.get("destination_zh", ""),
            "gt_verb": r["gt_verb"], "first_verb": r["first_verb"],
            "heading": r.get("heading"),
            "direction_pass": r.get("direction_pass"),
        }

    qwen_rows = [_to_qwen_row(r) for r in rows]

    # split
    if args.holdout_video:
        test = [r for r in qwen_rows if r["video"] == args.holdout_video]
        rest = [r for r in qwen_rows if r["video"] != args.holdout_video]
        rng = random.Random(args.seed)
        rng.shuffle(rest)
        n_val = max(1, int(len(rest) * args.val_frac / (1 - args.test_frac)))
        val, train = rest[:n_val], rest[n_val:]
        split_kind = f"video-holdout: {args.holdout_video}"
    else:
        rng = random.Random(args.seed)
        rng.shuffle(qwen_rows)
        n = len(qwen_rows)
        n_test = int(n * args.test_frac)
        n_val = int(n * args.val_frac)
        test = qwen_rows[:n_test]
        val = qwen_rows[n_test:n_test + n_val]
        train = qwen_rows[n_test + n_val:]
        split_kind = (f"per-variant random 80/10/10 "
                      f"(seed={args.seed}, variant={args.variant})")

    print(f"[to_sft] split: {split_kind}")
    print(f"  train: {len(train):,}")
    print(f"  val:   {len(val):,}")
    print(f"  test:  {len(test):,}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split_name, split_rows in (("train", train), ("val", val),
                                    ("test", test)):
        out = out_dir / f"a2_{args.variant}_{split_name}{args.suffix}.jsonl"
        with out.open("w", encoding="utf-8") as f:
            for r in split_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  wrote {out}  ({len(split_rows)} rows)")


if __name__ == "__main__":
    main()
