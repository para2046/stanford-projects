"""STEP 3 of the per-photo intersection match.

Compare LIST_A (from GPS_GEO.jsonl) with LIST_B (from VLM_GEO.jsonl)
per frame. A frame matches iff AT LEAST ONE name from each side
coincides — generous matching, no neighborhood-distance gate.

What counts as "coincides" (allows naming variance + affiliation):

  TYPE 1  exact      — fold(a) == fold(b)
                        e.g. "Grossmünster" == "grossmunster"
  TYPE 2  substring  — fold(a) contains fold(b) or vice-versa
                        e.g. "Grossmünster" ⊂ "Grossmünsterplatz"
                        e.g. "Limmat" ⊂ "Limmatquai"
                        e.g. "St. Peter" ⊂ "St. Peter Church"
  TYPE 3  word_share — both names share a meaningful word (≥4 chars)
                        e.g. "Hotel Storchen" and "Storchengasse"
                          share "storchen"
                        e.g. "Stadthausquai" and "Stadthaus"
                          share "stadthaus"

LIST_A = flat union of GPS_GEO.{attractions, landmarks, pois} names
LIST_B = flat union of VLM_GEO.{attractions, landmarks, pois} names

Tracking: each candidate name carries its source level so we can
later weight matches by strength (attraction-level > landmark > poi).

Output: data/cities/zurich/a2/GPS_VLM_GEO.jsonl
  one row per frame appearing in EITHER GPS_GEO or VLM_GEO:
    { video, frame_id,
      has_gps_geo: bool,
      has_vlm_geo: bool,
      list_a_gps:  { attractions: [...], landmarks: [...], pois: [...] }
      list_b_vlm:  { attractions: [...], landmarks: [...], pois: [...] }
      matches: [ { gps_name, gps_level,
                   vlm_name, vlm_level,
                   match_type } ]                      # all coincidences
      matched: bool                                    # any match exists
      best_level: "attraction" | "landmark" | "poi" | None
                                                       # strongest matched level
    }

  python -m src.a2_step3_gps_vlm_geo
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config                                       # noqa: E402


# Words too short or too common to count as "shared meaningful word".
# (street suffixes, generic place words — would over-match if used as
# match signal. e.g. "strasse" appears in every street name.)
COMMON_WORDS = {
    "strasse", "gasse", "platz", "brucke", "weg", "quai", "berg",
    "hof", "bahn", "haus", "kirche", "city", "stadt", "park", "fluss",
    "see", "kanal", "tor", "hauptbahnhof",   # too common alone
    "old", "new", "altstadt", "neustadt",
    "zurich", "zuerich", "swiss",            # always-on city/country names
    "the", "of", "and", "von", "der", "die", "das", "am", "an",
}


def fold(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s or "").lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def coincide_type(a: str, b: str):
    """Return ("exact"|"substring"|"word_share:tok"|None, detail).
    None = no coincidence."""
    fa, fb = fold(a), fold(b)
    if not fa or not fb:
        return None
    if fa == fb:
        return "exact"
    if fa in fb or fb in fa:
        return "substring"
    wa = {w for w in fa.split() if len(w) >= 4 and w not in COMMON_WORDS}
    wb = {w for w in fb.split() if len(w) >= 4 and w not in COMMON_WORDS}
    common = wa & wb
    if common:
        return f"word_share:{','.join(sorted(common))}"
    return None


def build_candidates(geo_row, vlm_row):
    """Return (list_a, list_b) where each is a list of (name, level)
    with level in {'attraction','landmark','poi'}.

    Dedup by lowercased name within each list (so a name appearing in
    both attractions and landmarks counts once, at the strongest level).
    """
    def _collect(row, keys):
        if not row:
            return []
        out = []
        seen = set()
        # process in priority order — attraction wins over landmark
        # wins over poi if a name is in multiple keys
        for key, level in keys:
            for entry in (row.get(key) or []):
                name = entry.get("name", "")
                f = fold(name)
                if not f or f in seen:
                    continue
                seen.add(f)
                out.append((name, level))
        return out

    a = _collect(geo_row, [
        ("attractions_within_R", "attraction"),
        ("landmarks_within_R", "landmark"),
        ("pois_within_R", "poi"),
    ])
    b = _collect(vlm_row, [
        ("attractions_from_vlm", "attraction"),
        ("landmarks_from_vlm", "landmark"),
        ("pois_from_vlm", "poi"),
    ])
    return a, b


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--gps-geo",
                    default=str(config.CITY_DIR / "a2"
                                / "GPS_GEO.jsonl"))
    ap.add_argument("--vlm-geo",
                    default=str(config.CITY_DIR / "a2"
                                / "VLM_GEO.jsonl"))
    ap.add_argument("--out",
                    default=str(config.CITY_DIR / "a2"
                                / "GPS_VLM_GEO.jsonl"))
    ap.add_argument("--cos-min", type=float, default=0.75,
                    help="DINOv2 cosine threshold — only frames whose "
                         "gps_recovery row has s_dino >= this are kept "
                         "(default 0.75, the cleaner cohort)")
    ap.add_argument("--gps-recovery",
                    default=str(config.CITY_DIR
                                / "gps_recovery_full.jsonl"),
                    help="source for s_dino lookup")
    ap.add_argument("--drop-ambiguous-heading", action="store_true",
                    help="drop frames whose heading_v2 decision is "
                         "'ambiguous' (gap ≤ 0.05). Reads "
                         "data/cities/zurich/a2/heading_v2.jsonl")
    args = ap.parse_args()

    # ── build s_dino lookup, then load both inputs ─────────────────
    sdino = {}
    for line in Path(args.gps_recovery).open(encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        sdino[(r["video"], r["frame_id"])] = r.get("s_dino", 0.0)

    def _pass_cos(k):
        return sdino.get(k, 0.0) >= args.cos_min

    # ── load heading_v2 decisions if dropping ambiguous ─────────────
    heading_decision = {}
    if args.drop_ambiguous_heading:
        hv2_path = config.CITY_DIR / "a2" / "heading_v2.jsonl"
        if not hv2_path.exists():
            sys.exit(f"--drop-ambiguous-heading set but "
                     f"{hv2_path} not found; "
                     f"run `python -m src.a2_heading_v2` first")
        for line in hv2_path.open(encoding="utf-8"):
            if not line.strip():
                continue
            d = json.loads(line)
            heading_decision[(d["video"], d["frame_id"])] = d["decision"]
        print(f"[step3] heading_v2 decisions loaded: "
              f"{len(heading_decision):,}")

    def _pass_heading(k):
        if not args.drop_ambiguous_heading:
            return True
        # keep only top1 or top1+top2; drop ambiguous + frames with no
        # heading decision row at all
        return heading_decision.get(k, "missing") in (
            "top1", "top1+top2")

    def _pass(k):
        return _pass_cos(k) and _pass_heading(k)

    gps_all = {}
    for line in Path(args.gps_geo).open(encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        gps_all[(r["video"], r["frame_id"])] = r
    vlm_all = {}
    for line in Path(args.vlm_geo).open(encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        vlm_all[(r["video"], r["frame_id"])] = r

    print(f"[step3] GPS_GEO frames (raw):                  "
          f"{len(gps_all):,}")
    print(f"[step3] VLM_GEO frames (raw):                  "
          f"{len(vlm_all):,}")
    print(f"[step3] filter: keep only frames with s_dino "
          f">= {args.cos_min}")

    gps = {k: v for k, v in gps_all.items() if _pass(k)}
    vlm = {k: v for k, v in vlm_all.items() if _pass(k)}
    if args.drop_ambiguous_heading:
        print(f"[step3] also dropping frames with ambiguous heading")
    print(f"[step3] GPS_GEO frames AFTER cos filter:       "
          f"{len(gps):,}")
    print(f"[step3] VLM_GEO frames AFTER cos filter:       "
          f"{len(vlm):,}")

    all_keys = set(gps) | set(vlm)
    overlap = set(gps) & set(vlm)
    print(f"[step3] union of frames (either side):         "
          f"{len(all_keys):,}")
    print(f"[step3] frames in BOTH (matchable):            "
          f"{len(overlap):,}")

    # ── compute match per frame ────────────────────────────────────
    LEVEL_RANK = {"attraction": 0, "landmark": 1, "poi": 2}
    LEVEL_LABEL = {0: "attraction", 1: "landmark", 2: "poi"}

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_matched = 0
    by_best_level = collections.Counter()
    n_match_types = collections.Counter()
    only_gps = 0
    only_vlm = 0
    both_no_match = 0

    with out_path.open("w", encoding="utf-8") as fout:
        for key in sorted(all_keys):
            geo_row = gps.get(key)
            vlm_row = vlm.get(key)
            list_a, list_b = build_candidates(geo_row, vlm_row)

            matches = []
            best_rank = None
            for ga, ga_lvl in list_a:
                for bv, bv_lvl in list_b:
                    t = coincide_type(ga, bv)
                    if t:
                        matches.append({
                            "gps_name": ga, "gps_level": ga_lvl,
                            "vlm_name": bv, "vlm_level": bv_lvl,
                            "match_type": t,
                        })
                        # match strength = stronger of the two levels
                        rk = min(LEVEL_RANK[ga_lvl],
                                  LEVEL_RANK[bv_lvl])
                        if best_rank is None or rk < best_rank:
                            best_rank = rk

            matched = bool(matches)
            best_level = (LEVEL_LABEL[best_rank]
                           if best_rank is not None else None)
            if matched:
                n_matched += 1
                by_best_level[best_level] += 1
                for m in matches:
                    n_match_types[m["match_type"].split(":")[0]] += 1
            else:
                if geo_row and not vlm_row:
                    only_gps += 1
                elif vlm_row and not geo_row:
                    only_vlm += 1
                else:
                    both_no_match += 1

            def _condense(row, keys):
                if not row:
                    return {}
                return {k.split("_")[0]: [e["name"]
                                            for e in (row.get(k) or [])]
                         for k in keys}

            row_out = {
                "video": key[0], "frame_id": key[1],
                "has_gps_geo": geo_row is not None,
                "has_vlm_geo": vlm_row is not None,
                "list_a_gps": _condense(geo_row, [
                    "attractions_within_R",
                    "landmarks_within_R",
                    "pois_within_R"]),
                "list_b_vlm": _condense(vlm_row, [
                    "attractions_from_vlm",
                    "landmarks_from_vlm",
                    "pois_from_vlm"]),
                "matches": matches,
                "matched": matched,
                "best_level": best_level,
            }
            fout.write(json.dumps(row_out, ensure_ascii=False) + "\n")

    print(f"[step3] wrote {out_path}")

    # ── summary ────────────────────────────────────────────────────
    print()
    print("=" * 96)
    print("RESULT")
    print("=" * 96)
    n = len(all_keys)
    print(f"frames total (union):                         {n:,}")
    print(f"  MATCHED (≥1 coincidence):                   "
          f"{n_matched:,}  ({100*n_matched/n:.1f} %)")
    print(f"     by best match level:")
    for lvl in ["attraction", "landmark", "poi"]:
        c = by_best_level[lvl]
        print(f"        {lvl:<11s}  {c:>5d}  "
              f"({100*c/max(1,n_matched):.1f} % of matched)")
    print(f"  unmatched, only in GPS_GEO:                 {only_gps:,}")
    print(f"  unmatched, only in VLM_GEO:                 {only_vlm:,}")
    print(f"  unmatched, in both but no coincidence:      {both_no_match:,}")

    print()
    print("match-type breakdown (across all matches):")
    for t, c in n_match_types.most_common():
        print(f"  {t:<13s} {c:>6d}")

    # ── samples ────────────────────────────────────────────────────
    print()
    print("=" * 96)
    print("SAMPLES — 3 matches per best_level (open image_path to verify)")
    print("=" * 96)
    samples = {"attraction": [], "landmark": [], "poi": []}
    with out_path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["matched"] and len(samples[r["best_level"]]) < 3:
                samples[r["best_level"]].append(r)
            if all(len(v) >= 3 for v in samples.values()):
                break
    for lvl in ["attraction", "landmark", "poi"]:
        print(f"\n--- best_level = {lvl} ---")
        for r in samples[lvl]:
            ms = [(m["gps_name"], m["vlm_name"], m["match_type"])
                   for m in r["matches"][:3]]
            print(f"  {r['video']}/{r['frame_id']}")
            print(f"    GPS attractions: {r['list_a_gps'].get('attractions',[])}")
            print(f"    VLM attractions: {r['list_b_vlm'].get('attractions',[])}")
            print(f"    first 3 matches:")
            for gn, vn, t in ms:
                print(f"      [{t}]  GPS='{gn}'  ⇄  VLM='{vn}'")


if __name__ == "__main__":
    main()
