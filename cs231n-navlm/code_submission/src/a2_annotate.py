"""Base annotation generator for Attempt 2.

For each (frame, destination) pair in routes.jsonl, calls Gemini Pro
2.5 ONCE with the heading-given prompt (template A from §19 of
DEV_MANUAL_v2.md) + interactive-guide system prompt. Saves response
text + verb extraction + format/direction passes.

The output `annotations_a2_base.jsonl` is the base file from which
`src/a2_derive_variants.py` will produce the 3 trained variants
(heading-given, heading-derived, heading-implicit) via text transforms.

  python -m src.a2_annotate --limit 5            # smoke test
  python -m src.a2_annotate --limit 0            # full run (~3,657 pairs)
  python -m src.a2_annotate --resume             # skip already-done pairs
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config                                       # noqa: E402
from src.gemini_api import call_gemini              # noqa: E402
from src.a2_attraction_slots import (               # noqa: E402
    ATTRACTIONS_21, ALIASES,
)


COMPASS_NAMES = [
    (0, "north"), (22.5, "north-northeast"), (45, "northeast"),
    (67.5, "east-northeast"), (90, "east"), (112.5, "east-southeast"),
    (135, "southeast"), (157.5, "south-southeast"), (180, "south"),
    (202.5, "south-southwest"), (225, "southwest"),
    (247.5, "west-southwest"), (270, "west"), (292.5, "west-northwest"),
    (315, "northwest"), (337.5, "north-northwest"),
]


def compass(deg):
    if deg is None:
        return ""
    best = min(COMPASS_NAMES, key=lambda x: abs(((deg - x[0] + 540) % 360) - 180))
    return best[1]


SYSTEM_PROMPT_COMMON_HEAD = """You are a Zurich walking-tour guide speaking directly to a tourist who is looking at the photo right now. Help them take the next step.

Useful Zurich orientation facts you may rely on when reasoning:
- The Limmat river flows roughly south-to-north through central Zurich.
- Grossmünster (twin towers) sits on the EAST bank of the Limmat.
- Fraumünster (single tall spire with green roof) sits on the WEST bank, across from Grossmünster.
- St. Peter (largest clock face in Europe) is on the WEST bank a bit north of Fraumünster.
- Bahnhofstrasse runs roughly south-to-north: Hauptbahnhof at the NORTH end, Paradeplatz mid-way, Bürkliplatz / Lake Zurich at the SOUTH end.
- At midday in Zurich the sun sits in the SOUTH (slightly south-east in morning, south-west in afternoon).
- Tram tracks visible on a street tell you the street's axis (along the tracks).

Your reply has two parts: <thinking> for reasoning and <answer> for the spoken instruction.

<answer> is one sentence speaking DIRECTLY to the walker (use "you"), pointing to specific things they can see, then the action verb:
  "Can you see X?"  "Look at the X."  "Notice the X ahead."
Reference only landmarks from the "Visible landmarks" list. End with the action verb on its own short sentence.

The action verb must be EXACTLY one of:
  continue ahead    turn left    turn right    turn around

GOOD <answer> examples:
  "Can you see Münsterbrücke directly ahead? Turn around."
  "Look at the cathedral towers on your left. Turn left."
  "Notice Bahnhofstrasse with shop signs stretching ahead. Continue ahead."
  "There is no clear landmark in front of you. Turn right."

AVOID:
  - Naming places NOT in the Visible landmarks list.
  - Mentioning the destination by name unless it's also visible now.
  - Compass directions ("head north") — say "ahead", "to your right".
"""

# Per-variant <thinking> rules — picked by --variant
THINKING_RULE = {
    "given": """In <thinking>, write 1-2 short sentences reasoning from the GIVEN heading and the route's first-segment bearing to a verb. State both numerically (e.g., "I'm facing 95° (east); the route heads 270° (west), a 180° rotation, so turn around").

Output template:
<thinking>
[1-2 sentences using the given heading + route bearing]
</thinking>
<answer>
[visual context using only Visible landmarks]. [VERB].
</answer>""",

    "derived": """In <thinking>, walk through a 4-step derivation of the walker's heading from the photo, then close with the verb decision. The heading number is NOT in the prompt — derive it from visual cues.

Use the orientation facts above (Limmat axis, church positions, Bahnhofstrasse axis, sun) plus what you SEE in the photo. Concrete cues to look for:
  - shop signs / store names (Google them mentally to known addresses)
  - tram tracks: which direction do they run? where do they lead?
  - sun and shadow direction (sun south → shadows fall north)
  - recognisable buildings/spires and their known compass orientation
  - the Limmat river (south-to-north) or Lake Zurich (the lake is south of the old town)
  - street width and architecture (Bahnhofstrasse vs old-town alleys)

Output template (each step on its own line — STEP-N: prefix optional but helpful):

<thinking>
STEP 1 (visual cues): I can see [2-3 concrete things in the photo].
STEP 2 (apply geography): These cues indicate the camera is oriented such that [direction reasoning, e.g., "Grossmünster on the right means the Limmat is on my right, so I'm facing roughly south"].
STEP 3 (estimated heading): I estimate I'm facing X° (compass direction).
STEP 4 (route comparison): The route's first segment heads Y° — that's a [N°] rotation [direction], so [verb].
</thinking>
<answer>
[visual context using only Visible landmarks]. [VERB].
</answer>""",

    "implicit": """In <thinking>, walk through a 3-step PURELY VISUAL chain of reasoning. Do NOT mention any numeric heading (neither given nor estimated). Locate the destination relative to the camera using only what's visible, then state the verb.

Concrete cues to look for:
  - visible landmarks from the "Visible landmarks" list — where are they in the frame? (centre? left edge? behind via reflection? not visible at all?)
  - tram tracks: do they go toward or away from the camera?
  - the Limmat river or Lake Zurich if visible — which side of the frame?
  - the destination itself — is any of it visible?

Output template (each step on its own line — STEP-N: prefix optional):

<thinking>
STEP 1 (what I see): The visible scene contains [list 2-3 specific things].
STEP 2 (where the destination is relative to me): [The destination is "ahead of me", "behind me", "to my left", "to my right", based on cues like "I see X to my left which is between me and the destination"]. NO numeric heading.
STEP 3 (verb decision): Therefore the walker should [verb].
</thinking>
<answer>
[visual context using only Visible landmarks]. [VERB].
</answer>""",
}


def system_prompt(variant: str, strip_visible: bool = False) -> str:
    """If strip_visible=True (no Visible-landmarks list in the user prompt),
    rewrite the two lines that reference the list so the system prompt
    stays consistent with what the user prompt actually contains."""
    head = SYSTEM_PROMPT_COMMON_HEAD
    if strip_visible:
        head = head.replace(
            "Reference only landmarks from the \"Visible landmarks\" list.",
            "Reference only landmarks you can directly see in the image.")
        head = head.replace(
            "  - Naming places NOT in the Visible landmarks list.",
            "  - Naming places NOT visible in the image (no hallucinated landmarks).")
    return head + "\n" + THINKING_RULE[variant]


def _shared_body(route_row, visible_landmarks):
    edge_bearing = route_row["route_bearing_network"]
    dist = route_row["route_distance_m"]
    first_seg = route_row["first_segment_length_m"]
    n_seg = route_row["n_segments"]
    dest_en = route_row["destination"]
    dest_zh = route_row.get("destination_zh", "")
    further = (f", then {n_seg-1} more turn{'s' if n_seg > 2 else ''} "
                f"over a total of {dist:.0f} m") if n_seg > 1 else ""
    visible_str = ", ".join(visible_landmarks) if visible_landmarks \
                   else "(no notable landmarks listed)"
    return (
        f"Destination: {dest_en} ({dest_zh}), about {dist:.0f} m walking "
        f"distance.\n\n"
        f"OSM walking route:\n"
        f"  First segment heads {edge_bearing:.0f}° ({compass(edge_bearing)}) "
        f"for {first_seg:.0f} m{further}.\n\n"
        f"Visible landmarks at this spot:\n  {visible_str}\n\n"
    )


def build_teacher_prompt(route_row, visible_landmarks, variant):
    """Prompt sent to Gemini — ALWAYS includes heading so the teacher
    can compute the correct verb. The CoT style is controlled by the
    variant-specific instruction appended after the shared body."""
    heading = route_row["heading"]
    body = _shared_body(route_row, visible_landmarks)
    base = (
        f"You are at this location, facing {heading:.0f}° "
        f"({compass(heading)}).\n\n" + body)
    if variant == "given":
        return base + (
            "Decide the next action verb. In <thinking>, REASON USING the "
            "given heading number and the route's first-segment bearing.")
    elif variant == "derived":
        return base + (
            "Decide the next action verb. In <thinking>, WRITE AS IF you "
            "derived the heading from the photo: start with "
            "\"I estimate I'm facing X° (direction).\" using the GIVEN "
            "heading value X. Then reason about the route and verb. "
            "Cite visual cues from the photo that support the heading "
            "estimate (shop signs, tram direction, sun position, "
            "recognisable buildings).")
    elif variant == "implicit":
        return base + (
            "Decide the next action verb. In <thinking>, write 1-2 short "
            "sentences of PURELY VISUAL reasoning. Do NOT mention any "
            "numeric heading (neither given nor estimated). Describe where "
            "the destination is in the photo using visible cues, then "
            "state the verb.")
    raise ValueError(variant)


def build_student_prompt(route_row, visible_landmarks, variant):
    """Prompt the STUDENT model will see at training and inference.
    Heading is hidden for derived/implicit variants."""
    heading = route_row["heading"]
    body = _shared_body(route_row, visible_landmarks)
    if variant == "given":
        return (
            f"You are at this location, facing {heading:.0f}° "
            f"({compass(heading)}).\n\n" + body
            + "Decide the next action verb.")
    elif variant == "derived":
        return (
            body
            + "The walker's heading is NOT provided. In <thinking>, FIRST "
              "infer the heading from the photo by stating "
              "\"I estimate I'm facing X° (direction)\", THEN reason about "
              "the route and verb.")
    elif variant == "implicit":
        return (
            body
            + "The walker's heading is NOT provided. Reason from visual cues "
              "in the photo about where the destination is and which verb is "
              "needed. Do NOT state a numeric heading.")
    raise ValueError(variant)


VERBS = ("continue ahead", "turn left", "turn right", "turn around")


def parse_answer(text):
    """Returns dict with format_pass, first_verb, anchors, etc.
    Truncation-robust: if </thinking> or </answer> is missing (model
    hit MAX_TOKENS mid-stream), we still extract what we can."""
    out = {"format_pass": False, "first_verb": None,
           "answer_text": "", "thinking_text": "",
           "truncated": False}
    # thinking — handle missing </thinking>
    t_open = text.find("<thinking>")
    a_open = text.find("<answer>")
    if t_open >= 0:
        t_close = text.find("</thinking>", t_open)
        if t_close >= 0:
            out["thinking_text"] = text[t_open + 10:t_close].strip()
        elif a_open > t_open:
            # </thinking> missing but <answer> exists — thinking
            # text is between <thinking> and <answer>
            out["thinking_text"] = text[t_open + 10:a_open].strip()
        else:
            # both close tags missing — thinking text is the rest
            out["thinking_text"] = text[t_open + 10:].strip()
            out["truncated"] = True
    # answer — handle missing </answer>
    # Pro 2.5 typically OMITS the </answer> closing tag — it stops after
    # the verb. We treat this as VALID format (the answer is complete,
    # just unclosed). 'truncated' is only set when <answer> itself is
    # missing (real truncation mid-thinking).
    if a_open >= 0:
        a_close = text.find("</answer>", a_open)
        if a_close >= 0:
            out["answer_text"] = text[a_open + 8:a_close].strip()
        else:
            # </answer> missing — take rest of text after <answer>.
            # Still considered valid format.
            out["answer_text"] = text[a_open + 8:].strip()
    elif out["thinking_text"]:
        # no <answer> tag at all → fall back to thinking text and
        # flag as truncated (real mid-stream cutoff)
        out["answer_text"] = out["thinking_text"]
        out["truncated"] = True

    # first verb (longest-match first, case-insensitive)
    if not out["answer_text"]:
        return out
    earliest = float("inf"); first = None
    for v in sorted(VERBS, key=len, reverse=True):
        m = re.search(rf"\b{re.escape(v)}\b", out["answer_text"], re.I)
        if m and m.start() < earliest:
            first, earliest = v, m.start()
    out["first_verb"] = first
    # format_pass: opened tags + valid first verb + not truncated.
    out["format_pass"] = (
        t_open >= 0 and a_open >= 0
        and first is not None and not out["truncated"])
    return out


def derived_heading_from_thinking(thinking_text):
    """For variant='derived' — try to extract the model's claimed
    heading from a sentence like 'I estimate I'm facing 95° (east)'.
    Returns float degrees or None."""
    m = re.search(
        r"facing\s+(\d{1,3}(?:\.\d+)?)\s*°",
        thinking_text, re.I)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--routes",
                    default=str(config.CITY_DIR / "a2" / "routes.jsonl"))
    ap.add_argument("--vlm-geo",
                    default=str(config.CITY_DIR / "a2" / "VLM_GEO.jsonl"))
    ap.add_argument("--variant", choices=["given", "derived", "implicit"],
                    default="given",
                    help="which user-prompt template + thinking-style: "
                         "given (heading in prompt; CoT uses it), "
                         "derived (no heading; CoT derives it from image), "
                         "implicit (no heading; CoT reasons visually)")
    ap.add_argument("--output", default=None,
                    help="output jsonl path; default = "
                         "data/cities/zurich/a2/annotations_a2_{variant}.jsonl")
    ap.add_argument("--limit", type=int, default=5,
                    help="cap to N pairs (5 = smoke test; 0 = all)")
    ap.add_argument("--resume", action="store_true",
                    help="skip pairs already in --output")
    ap.add_argument("--model", default="gemini-2.5-pro")
    ap.add_argument("--max-tokens", type=int, default=4096,
                    help="cap teacher tokens — Pro 2.5 burns hidden "
                         "thinking tokens against this budget, so set "
                         "high (4096 typ; the visible response is only "
                         "2-3 sentences)")
    args = ap.parse_args()

    # default output by variant
    if args.output is None:
        args.output = str(config.CITY_DIR / "a2"
                          / f"annotations_a2_{args.variant}.jsonl")

    # load routes
    routes = [json.loads(l) for l in
              Path(args.routes).open(encoding="utf-8") if l.strip()]
    print(f"[a2_annotate] variant: {args.variant}", flush=True)
    print(f"[a2_annotate] routes loaded: {len(routes):,}", flush=True)
    print(f"[a2_annotate] output: {args.output}", flush=True)
    print(f"[a2_annotate] GCP_PROJECT (env or config): "
          f"{os.environ.get('GCP_PROJECT') or config.GCP_PROJECT}",
          flush=True)

    # load VLM_GEO for per-frame Visible landmarks
    vlm = {}
    for line in Path(args.vlm_geo).open(encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        vlm[(r["video"], r["frame_id"])] = r

    # resume support
    done = set()
    out_path = Path(args.output)
    if args.resume and out_path.exists():
        for line in out_path.open(encoding="utf-8"):
            if not line.strip():
                continue
            d = json.loads(line)
            done.add((d["video"], d["frame_id"], d["destination"]))
        print(f"[a2_annotate] already done: {len(done):,}  (--resume)",
              flush=True)

    todo = [r for r in routes
            if (r["video"], r["frame_id"], r["destination"]) not in done]
    if args.limit > 0:
        todo = todo[:args.limit]
    print(f"[a2_annotate] processing: {len(todo):,}", flush=True)

    if not todo:
        print("[a2_annotate] nothing to do, exiting", flush=True)
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume else "w"
    t0 = time.time()
    n_kept = n_fmt_fail = n_dir_fail = 0

    with out_path.open(mode, encoding="utf-8") as fout:
        for i, r in enumerate(todo, 1):
            key = (r["video"], r["frame_id"])
            vrow = vlm.get(key, {})
            # visible landmarks for this frame — from VLM_GEO
            vis = sorted({e["name"] for e in
                            (vrow.get("attractions_from_vlm") or [])})

            teacher_msg = build_teacher_prompt(r, vis, args.variant)
            student_msg = build_student_prompt(r, vis, args.variant)
            sys_msg = system_prompt(args.variant)
            frame_path = (config.FRAMES_DIR / r["video"]
                          / f"{r['frame_id']}.jpg")

            try:
                resp = call_gemini(
                    str(frame_path), sys_msg, teacher_msg,
                    model=args.model, max_tokens=args.max_tokens,
                    label=f"a2_annot_{args.variant}_{r['video']}_"
                          f"{r['frame_id']}_{r['destination'][:8]}")
            except Exception as e:
                print(f"  ERROR {key} → {r['destination']}: "
                      f"{type(e).__name__}: {e}", flush=True)
                continue

            parsed = parse_answer(resp)
            direction_pass = (parsed["first_verb"] == r["gt_verb"])
            row = {
                "video": r["video"], "frame_id": r["frame_id"],
                "destination": r["destination"],
                "destination_zh": r.get("destination_zh", ""),
                "teacher_prompt": teacher_msg,
                "student_prompt": student_msg,
                "response": resp,
                "thinking": parsed["thinking_text"],
                "answer": parsed["answer_text"],
                "first_verb": parsed["first_verb"],
                "gt_verb": r["gt_verb"],
                "format_pass": parsed["format_pass"],
                "direction_pass": direction_pass,
                "PASS": parsed["format_pass"] and direction_pass,
                "truncated": parsed["truncated"],
                "variant": args.variant,
                "derived_heading": (
                    derived_heading_from_thinking(parsed["thinking_text"])
                    if args.variant == "derived" else None),
                # carry over for downstream
                "heading": r["heading"],
                "route_bearing_network": r["route_bearing_network"],
                "route_distance_m": r["route_distance_m"],
                "first_segment_length_m": r["first_segment_length_m"],
                "n_segments": r["n_segments"],
                "sampling_band": r["sampling_band"],
                "visible_landmarks": vis,
            }
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            fout.flush()
            n_kept += 1
            if not parsed["format_pass"]:
                n_fmt_fail += 1
            elif not direction_pass:
                n_dir_fail += 1

            if i % 25 == 0 or i == len(todo):
                elapsed = time.time() - t0
                rate = i / elapsed
                eta_s = (len(todo) - i) / rate if rate > 0 else 0
                cost = i * 0.008
                print(f"  [{i:4d}/{len(todo)}] kept={n_kept} "
                      f"fmt_fail={n_fmt_fail} dir_fail={n_dir_fail} "
                      f"~${cost:.2f} ETA {eta_s/60:.0f} min",
                      flush=True)

    print(flush=True)
    print(f"[a2_annotate] === SUMMARY ===")
    print(f"  pairs processed:   {len(todo):,}")
    print(f"  kept:              {n_kept:,}")
    print(f"  format failures:   {n_fmt_fail}")
    print(f"  direction failures: {n_dir_fail}")
    elapsed = time.time() - t0
    print(f"  wall time:         {elapsed/60:.1f} min "
          f"({elapsed/max(1,len(todo)):.1f} s/pair)")
    print(f"  est cost:          ${len(todo)*0.008:.2f}")
    print(f"  written to:        {out_path}")


if __name__ == "__main__":
    main()
