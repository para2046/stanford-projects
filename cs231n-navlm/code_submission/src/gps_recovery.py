"""Stage 6 — GPS recovery for video frames (DEV_MANUAL §2.5).

For each video frame:
  1. DINOv2 embed -> cosine-match against the Street View index (top-k)
  2. VLM (Gemini Pro) names the place -> resolved to GPS
  3. src.reconcile combines the two into an accept/reject + GPS
  4. heading = circular mean of the matched crops' rendered headings

Pure functions (cosine_topk, circular_mean) are unit-tested; embedding
needs DINOv2/GPU and the VLM step needs a GEMINI_API_KEY.

    python -m src.gps_recovery        # runs the full stage (heavy)
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config                       # noqa: E402
from src import reconcile           # noqa: E402


def cosine_topk(query_emb, ref_embs, k=config.DINOV2_TOPK):
    """Top-k reference indices + cosine similarities for one query.

    query_emb: (D,) ; ref_embs: (N, D). Both assumed L2-normalised, so
    cosine = dot product. Returns (indices, sims), best-first.
    """
    import numpy as np
    sims = ref_embs @ query_emb
    order = np.argsort(-sims)[:k]
    return order, sims[order]


def circular_mean(degrees, weights=None):
    """Circular mean of headings (degrees), in [0, 360). Optional
    per-angle weights — used to compute the cosine-weighted mean of
    the four crops at one pano (so a walker heading 45° between the
    pano's 0° and 90° crops gets ≈ 45°, not 0° or 90°)."""
    if not degrees:
        return None
    if weights is None:
        weights = [1.0] * len(degrees)
    x = sum(w * math.sin(math.radians(d)) for w, d in zip(weights, degrees))
    y = sum(w * math.cos(math.radians(d)) for w, d in zip(weights, degrees))
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def circular_spread(degrees, weights=None):
    """Circular standard deviation (degrees) — heading confidence
    proxy. With weights, divides by Σweights so a 4-way tie (any pair
    of opposite directions weighted equally) returns 360° (fully
    ambiguous), and a clean single direction returns ≈ 0°."""
    if not degrees:
        return 360.0
    if weights is None:
        weights = [1.0] * len(degrees)
    W = sum(weights)
    if W < 1e-9:
        return 360.0
    x = sum(w * math.sin(math.radians(d))
            for w, d in zip(weights, degrees)) / W
    y = sum(w * math.cos(math.radians(d))
            for w, d in zip(weights, degrees)) / W
    r = math.sqrt(x * x + y * y)
    if r >= 0.9999:
        return 0.0
    if r < 1e-9:                       # fully dispersed -> max spread
        return 360.0
    return math.degrees(math.sqrt(-2 * math.log(r)))


def recover_frame(query_emb, ref):
    """Recover (gps, heading) for one frame.

    ref: dict with embs (N,D), gps [(lat,lon)...], headings [deg...],
         and `vlm` = a callable frame->(place_gps, confidence).
    Returns a reconcile result dict + heading fields.
    """
    idx, sims = cosine_topk(query_emb, ref["embs"])
    dino_gps = ref["gps"][int(idx[0])]
    heading = circular_mean([ref["headings"][int(i)] for i in idx])
    spread = circular_spread([ref["headings"][int(i)] for i in idx])

    vlm_gps, vlm_conf = ref["vlm"]()           # VLM place-naming
    result = reconcile.reconcile(dino_gps, float(sims[0]), vlm_gps, vlm_conf)
    result["heading"] = heading
    result["heading_spread"] = spread
    return result


def main():
    """End-to-end GPS recovery for ALL extracted frames (DEV_MANUAL
    §2.5, *GPS recovery for all*).

    Iterates every frame in the DINOv2 frame cache (typically the
    `frames_n1_l0.npz` covering all ~22k extracted frames). For each
    frame two acceptance tiers are possible:

      tier 1 — frame is in `poi_scan.jsonl` (the every-30 VLM sample)
        F1 cos_dino >= MIN_SIM
        F2 vlm_gps  is not None
        F3 exact-name match OR distance(vlm_POI, dino_nearest_POI)
                                       <= NEIGHBORHOOD_RADIUS_M
        gps = g_dino  (the SV pano's known coordinates)
        -> highest-confidence label, the "trustworthy" set the
        annotator will use directly.

      tier 2 — frame has no VLM signal (the other ~21k frames)
        F1 cos_dino >= MIN_SIM
        gps = g_dino
        -> "DINOv2 candidate"; weaker per-frame confidence but
        useful in bulk. HMM map-matching (§2.5 next stage) uses
        route continuity to filter out single-frame errors.

    Each tier records its own `accepted` / `reject_reason`. Headings
    (cosine-weighted same-pano mean) + heading_gap are computed for
    every frame regardless of tier.
    """
    import argparse
    import collections
    import json
    import numpy as np
    from tqdm import tqdm
    from src import reconcile
    from src.geo_check import geo_check_from_scan
    from src.spatial import (build_poi_index, nearest_poi_m,
                              name_matches_poi, distance_pois_m)

    ap = argparse.ArgumentParser(
        description="GPS recovery — DINOv2 + VLM, weighted reconcile")
    ap.add_argument("-k", type=int, default=config.DINOV2_TOPK,
                    help="top-K SV matches per frame")
    ap.add_argument("--min-sim", type=float, default=config.MIN_SIM,
                    help="reject DINOv2 match below this cosine "
                         "(filter F1, default config.MIN_SIM)")
    ap.add_argument("--max-var-m", type=float,
                    default=config.RECONCILE_MAX_VAR_M,
                    help="reject if |g_dino - g_vlm| > this in metres "
                         "(filter F3, default config.RECONCILE_MAX_VAR_M)")
    ap.add_argument("--frame-cache", type=str, default="frames_n1_l0",
                    help="DINOv2 frame cache name (default frames_n1_l0 "
                         "= all extracted frames; use frames_n30_l0 to "
                         "rerun only the every-30 VLM-sampled set)")
    ap.add_argument("--output", type=str, default="gps_recovery_all.jsonl",
                    help="output filename under data/cities/zurich/ "
                         "(default keeps the pilot 'gps_recovery.jsonl' "
                         "untouched)")
    ap.add_argument("--poi-scan", type=str, default="poi_scan.jsonl",
                    help="POI-scan jsonl that defines the tier-1 frames "
                         "(default poi_scan.jsonl, the every-30 sample; "
                         "use poi_scan_cos0.75.jsonl to fold in the "
                         "cos>=0.75 VLM expansion — promotes ~4k tier-2 "
                         "DINOv2-only frames into tier-1 candidates "
                         "subject to F1/F2/F3)")
    args = ap.parse_args()

    # OSM POIs — VLM guess -> GPS lookup + spatial index for the
    # semantic cross-check (nearest POI to DINOv2's SV pano).
    pois = json.loads(
        (config.CITY_DIR / "pois.json").read_text(encoding="utf-8"))
    pois_map = {p["name"]: p for p in pois}
    poi_index = build_poi_index(pois)
    print(f"[gps_recovery] indexed {len(poi_index['pois'])} / {len(pois)} "
          f"POI geometries for nearest-POI cross-check")

    # SV per-image meta — id -> {lat, lon, heading, pano_id}
    sv_meta_path = config.STREETVIEW_DIR / "meta.jsonl"
    if not sv_meta_path.exists():
        sys.exit(f"no SV meta at {sv_meta_path} — copy it from the v1 "
                 f"streetview folder.")
    sv_meta = {}
    for line in sv_meta_path.open(encoding="utf-8"):
        m = json.loads(line)
        sv_meta[m["id"]] = {
            "lat": m["lat"], "lon": m["lon"],
            "heading": m.get("compass_angle", 0),
            "pano_id": m.get("pano_id", ""),
        }

    # DINOv2 caches (sv refs + video frames)
    cdir = config.CITY_DIR / "dinov2"
    sv_cache = np.load(cdir / "sv_v1.npz", allow_pickle=True)
    sv_embs = sv_cache["embs"]
    sv_ids = [Path(p).stem for p in sv_cache["paths"]]   # image filename stem

    fpath = cdir / f"{args.frame_cache}.npz"
    if not fpath.exists():
        sys.exit(f"no DINOv2 frame cache at {fpath} — "
                 f"run `python -m src.dinov2_match --every-n 30` first")
    fcache = np.load(fpath, allow_pickle=True)
    frame_embs = fcache["embs"]
    frame_paths = [Path(p) for p in fcache["paths"]]
    # (video, frame_id) -> index into frame_embs
    frame_idx = {(p.parent.name, p.stem): i
                 for i, p in enumerate(frame_paths)}

    # pre-group SV crops by pano_id so we can compute same-pano heading
    # confidence (A) per frame: of the 4 crops at top-1's pano, the
    # cosine gap between the best heading and the second tells us how
    # well DINOv2 can pin down direction (not just location).
    pano_id_per_crop = [sv_meta.get(sid, {}).get("pano_id", "")
                         for sid in sv_ids]
    pano_to_crop_idx = collections.defaultdict(list)
    for i, pid in enumerate(pano_id_per_crop):
        if pid:
            pano_to_crop_idx[pid].append(i)

    # POI-scan rows, indexed by (video, frame_id) for tier-1 lookup.
    # `--poi-scan` lets us swap in the cos>=0.75 expansion file —
    # promotes ~4 k tier-2 DINOv2 candidates into tier-1 (still gated
    # by F1+F2+F3, so promotion ≠ acceptance).
    scan_path = config.CITY_DIR / args.poi_scan
    if not scan_path.exists():
        sys.exit(f"poi-scan file not found: {scan_path}")
    scan_index = {}
    for line in scan_path.open(encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        scan_index[(r["video"], r["frame_id"])] = r
    print(f"[gps_recovery] poi-scan source:        {scan_path.name}")
    print(f"[gps_recovery] frames in DINOv2 cache: {len(frame_idx)}")
    print(f"[gps_recovery] VLM-sampled frames (tier-1 candidates): "
          f"{len(scan_index)}")

    out_path = config.CITY_DIR / args.output
    counts = collections.Counter()

    with out_path.open("w", encoding="utf-8") as fout:
        # iterate ALL frames in the cache, sorted for deterministic +
        # downstream-HMM-friendly ordering (per-video frame sequence).
        # Frames with cosine < MIN_SIM (or no SV-meta lookup) are
        # DROPPED at the top of the loop — they have no useful signal
        # and would only clutter the output / downstream viz.
        for (video, frame_id), i in tqdm(
                sorted(frame_idx.items()),
                desc="[gps_recovery]", unit="frame"):
            # quick F1 check up-front: drop weak DINOv2 matches before
            # doing all the per-pano work.
            sims_top1 = float(np.max(sv_embs @ frame_embs[i]))
            if sims_top1 < args.min_sim:
                counts["dropped_dino_weak"] += 1
                continue
            row = scan_index.get((video, frame_id))    # None for tier 2

            # ── DINOv2 ─────────────────────────────────────────────
            # full cosine vector against all SV crops -> top-K + the
            # SAME-PANO heading-confidence diagnostic
            sims_all = sv_embs @ frame_embs[i]
            top_idx = np.argpartition(-sims_all, args.k)[:args.k]
            top_idx = top_idx[np.argsort(-sims_all[top_idx])]
            top_sims = sims_all[top_idx]
            idx, sims = top_idx, top_sims          # legacy variable names

            top_id = sv_ids[int(idx[0])]
            top_meta = sv_meta.get(top_id, {})
            dino_loc = ((top_meta["lat"], top_meta["lon"],
                          top_meta["heading"])
                         if top_meta else None)
            s_dino = float(sims[0])

            # ── HEADING: from the 4 crops at top-1's PANO only ──
            # We trust one standard — the pano DINOv2 picked — and
            # interpolate direction from the cosines of *its* 4
            # compass crops. Mixing top-K across different panos
            # averages apples and oranges.
            top_pano = top_meta.get("pano_id", "")
            same_pano = pano_to_crop_idx.get(top_pano, [])
            same_pano_angles = [
                sv_meta[sv_ids[j]]["heading"] for j in same_pano]
            same_pano_weights = [
                max(float(sims_all[j]), 0.0)        # clamp negatives
                for j in same_pano]
            heading = circular_mean(same_pano_angles, same_pano_weights)
            spread = circular_spread(same_pano_angles, same_pano_weights)
            same_pano_heading_count = len(same_pano)

            # heading_gap: how much does the best heading beat the 2nd
            # at this same pano? high = well-defined; low = ambiguous.
            same_pano_sims_sorted = sorted(
                same_pano_weights, reverse=True)
            heading_gap = (
                ((same_pano_sims_sorted[0] - same_pano_sims_sorted[1])
                 / same_pano_sims_sorted[0])
                if len(same_pano_sims_sorted) >= 2
                   and same_pano_sims_sorted[0] > 0
                else 1.0)

            # ── VLM (only if scan row exists — tier 1) ───────────
            vlm = (geo_check_from_scan(row, pois_map) if row is not None
                   else {"gps": None, "confidence": "",
                         "place_name": "", "reasoning": ""})

            # ── semantic + neighborhood cross-checks (only meaningful
            #    when VLM data exists — tier 1).
            dino_nearest = None
            dino_nearest_m = None
            exact_name_match = False     # diagnostic
            neighborhood_match = False   # F3 gate
            poi_dist_m = None
            if dino_loc is not None:
                dino_nearest, dino_nearest_m = nearest_poi_m(
                    dino_loc[0], dino_loc[1], poi_index)
                if row is not None:        # only useful when we have a VLM
                    exact_name_match = name_matches_poi(
                        vlm["place_name"], dino_nearest)
                    if vlm["place_name"] and dino_nearest:
                        poi_dist_m = distance_pois_m(
                            vlm["place_name"], dino_nearest["name"],
                            poi_index)
                        neighborhood_match = (
                            poi_dist_m <= config.NEIGHBORHOOD_RADIUS_M)
            semantic_match = exact_name_match or neighborhood_match

            tier = 1 if row is not None else 2
            base = {
                "video": video, "frame_id": frame_id, "tier": tier,
                "g_dino": [dino_loc[0], dino_loc[1]] if dino_loc else None,
                "s_dino": s_dino,
                "g_vlm": list(vlm["gps"]) if vlm["gps"] else None,
                "vlm_conf": vlm["confidence"],
                "place_guess": vlm["place_name"],          # resolved OSM
                "vlm_guess_raw": row.get("guess", "") if row else "",
                "reasoning": vlm["reasoning"][:240] if row else "",
                "dino_nearest_name": (dino_nearest["name"]
                                       if dino_nearest else ""),
                "dino_nearest_m": dino_nearest_m,
                "poi_dist_m": poi_dist_m,
                "exact_name_match": exact_name_match,
                "neighborhood_match": neighborhood_match,
                "semantic_match": semantic_match,          # F3 gate
                "top_sv_id": top_id,
                "heading": heading, "heading_spread": spread,
                "heading_gap": heading_gap,
                "same_pano_heading_count": same_pano_heading_count,
            }

            # ── decision: tier 1 uses full F1/F2/F3; tier 2 is F1 only
            if dino_loc is None:
                # SV pano matched but its meta entry is missing
                # (shouldn't happen with a clean meta.jsonl); drop too.
                counts["dropped_no_sv_meta"] += 1
                continue
            if tier == 1:
                r = reconcile.reconcile_strict(
                    (dino_loc[0], dino_loc[1]), s_dino,
                    vlm["gps"], vlm["confidence"],
                    min_sim=args.min_sim, max_var_m=args.max_var_m,
                    semantic_match=semantic_match)
                base.update({
                    "accepted": r["accepted"], "score": r["score"],
                    "variance_m": r["variance_m"],
                    "spatial_match": r["spatial_match"],
                    "gps": list(r["gps"]) if r["gps"] else None,
                    "reject_reason": r["reject_reason"],
                })
                bucket = ("t1_accepted" if r["accepted"]
                          else f"t1_{r['reject_reason']}")
                counts[bucket] += 1
            else:                                       # tier 2 — DINOv2 only
                # F1 was already enforced at the top of the loop, so any
                # tier-2 frame reaching here passes F1 -> accepted.
                base.update({"accepted": True, "score": s_dino,
                             "variance_m": None,
                             "gps": [dino_loc[0], dino_loc[1]],
                             "reject_reason": "",
                             "spatial_match": None})
                counts["t2_accepted"] += 1

            fout.write(json.dumps(base, ensure_ascii=False) + "\n")
            fout.flush()

    print(f"[gps_recovery] wrote {out_path}")
    print(f"[gps_recovery] reconcile_strict | min_sim={args.min_sim} "
          f"| max_var_m={args.max_var_m} | "
          f"neighborhood_m={config.NEIGHBORHOOD_RADIUS_M}")
    total = sum(counts.values())
    written = total - counts.get("dropped_dino_weak", 0) \
                    - counts.get("dropped_no_sv_meta", 0)
    # tier breakdown
    order = ["t1_accepted", "t1_disagree", "t1_dino_weak",
             "t1_vlm_unresolved", "t1_no_sv_meta",
             "t2_accepted",
             "dropped_dino_weak", "dropped_no_sv_meta"]
    for k in order:
        v = counts.get(k, 0)
        if v:
            print(f"  {k:22s} {v:6d}  ({100 * v / max(1, total):4.1f}%)")
    print(f"  {'TOTAL FRAMES SEEN':22s} {total:6d}")
    print(f"  {'WRITTEN TO OUTPUT':22s} {written:6d}  "
          f"(rows in {out_path.name})")


if __name__ == "__main__":
    main()
