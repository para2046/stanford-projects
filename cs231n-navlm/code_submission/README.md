# NavLM v2 — Code Submission

**CS231n 2026 final project · author: para2046**

End-to-end pipeline for training a compass-free, landmark-free
vision-language navigation model on Zurich walking-tour videos. A
trained Qwen 2.5 VL 7B + LoRA adapter takes a single street-level
photo + a destination name and emits one of four egocentric verbs
(`continue ahead` / `turn left` / `turn right` / `turn around`).

> The headline result of this submission: a LoRA-fine-tuned model
> reaches **67.2 %** PASS on the `derived` variant (model must
> visually estimate its heading) and **62.2 %** on the `implicit`
> variant (purely visual, no numeric heading allowed), starting
> from zero-shot baselines of 30.2 % and 25.1 % respectively.

## Repository layout

```
code_submission/
├── README.md                 ← this file
├── requirements.txt          ← Python deps (see file for details)
├── config.py                 ← single source of truth for data paths
└── src/
    ├── __init__.py
    ├── download_videos.py    ← Stage 1 (video → frames)
    ├── extract_frames.py
    ├── streetview.py         ← Stage 2 (StreetView pano fetch + DINO)
    ├── dinov2_match.py
    ├── gps_recovery.py
    ├── geo_check.py
    ├── build_walking_graph.py ← Stage 3 (OSM walking graph + snap)
    ├── road_snap.py
    ├── spatial.py
    ├── routing.py
    ├── pois.py               ← Stage 4 (POIs + 21-attraction vocab)
    ├── poi_scan.py
    ├── a2_attraction_slots.py
    ├── a2_step1_gps_geo.py   ← Stage 5 (3-way GPS / VLM join + matching)
    ├── a2_step2_vlm_geo.py
    ├── a2_step3_gps_vlm_geo.py
    ├── a2_join_3way.py
    ├── a2_match_strict.py
    ├── a2_match_intersection.py
    ├── a2_proximity_tag.py
    ├── a2_heading_v2.py      ← Stage 6 (heading + target frames + routes)
    ├── a2_target_frames.py
    ├── a2_destination_targets.py
    ├── a2_route.py
    ├── a2_sv_pano_attractions.py
    ├── gemini_api.py         ← Stage 7 (Gemini Pro 2.5 teacher)
    ├── a2_annotate.py
    ├── a2_to_sft.py          ← Stage 8 (SFT conversion w/ --strip-visible)
    ├── a2_sanity_check.py
    ├── a2_train_modal.py     ← Stage 9 (LoRA SFT on Modal A100-80GB)
    ├── a2_eval_modal.py      ← Stage 10 (inference on Modal A100-40GB)
    ├── a2_score.py           ← local PASS scoring
    ├── a2_figures_nov.py     ← paper figures (fig 1, 2, 3, 3b, 3c)
    ├── a2_fig_map_vlm.py     ← paper figure 5 (map+VLM grounding)
    ├── a2_fig_match_examples.py ← paper figure 4 (DINOv2 match QC)
    ├── a2_viz_matched.py     ← interactive QC viewers
    ├── a2_viz_thin.py
    ├── a2_viz_sft.py
    └── a2_viz_route_gt.py
```

## Setup

```bash
# 1. Python environment (Python 3.10)
conda create -n navlm python=3.10 -y && conda activate navlm
pip install -r requirements.txt

# 2. Modal account (free credit at https://modal.com)
modal token new           # one-time browser auth

# 3. Google Cloud project for Gemini (one or more, for parallel annotation)
export GEMINI_API_KEY=<your-key-here>
# or set GOOGLE_APPLICATION_CREDENTIALS=<path-to-sa-key.json>

# 4. (optional) Google StreetView Static API key for Stage 2
export STREETVIEW_API_KEY=<your-key-here>

# 5. Data root — single env var controls every path in config.py.
#    Defaults to ./data if unset.
export NAVLM_DATA=/path/to/data    # POSIX
$env:NAVLM_DATA = "C:\path\to\data"   # PowerShell
```

All paths in `config.py` are derived from `NAVLM_DATA`; no script
hard-codes absolute paths.

## Pipeline order

The pipeline runs in 10 stages. Each stage's outputs become the
next stage's inputs. Every script is invoked as a module from the
project root: `python -m src.<module> [args...]`. Wall-times below
assume a workstation laptop for local steps; Modal stages list
A100-hours.

### Stage 1 — Video → frames

Download a curated set of 7 Zurich walking-tour videos and slice
them to 1 fps stills.

```bash
python -m src.download_videos                # ~12 min, ~1.5 GB
python -m src.extract_frames                 # ~8 min, ~5,000 frames/video
```

Outputs: `data/cities/zurich/videos/*.mp4`,
`data/cities/zurich/frames/<video>/frame_XXXXX.jpg`.

### Stage 2 — StreetView panorama crawl + DINOv2 matching

Fetch a city-block grid of StreetView panoramas (4 cardinal crops
per pano), then build a DINOv2-large embedding cache for both the
SV crops and the query frames.

```bash
python -m src.streetview                     # ~30 min, $20 in SV API credits
python -m src.dinov2_match                   # ~25 min on GPU, ~5 min on CPU
python -m src.gps_recovery                   # match query frames → SV pano + GPS
```

Outputs: `data/cities/streetview/zurich/images/*.jpg` (~4,400 crops),
`data/cities/zurich/dinov2/*.npz` (cached embeddings),
`data/cities/zurich/gps_recovery_full.jsonl` (15,053 accepted frames).

### Stage 3 — OSM walking graph + GPS snapping

```bash
python -m src.build_walking_graph            # ~3 min — fetches OSM via osmnx
python -m src.road_snap                      # HMM-snap raw GPS to walking graph
```

Outputs: `data/cities/zurich/osm_walking.pkl` (17,996 nodes / 48,218 edges),
`data/cities/zurich/road_snapped.jsonl`.

### Stage 4 — POIs + 21-attraction vocabulary

```bash
python -m src.pois                           # extract POIs from OSM
python -m src.poi_scan                       # filter to navigation-worthy POIs
```

Output: `data/cities/zurich/pois.json` (~1,300 POIs),
`data/cities/zurich/a2/attraction_slots.jsonl` for the 21-attraction
vocabulary used by every downstream stage.

### Stage 5 — 3-way GPS / VLM join

Use Gemini Vision to scan each accepted frame for visible POIs,
then intersect those names with the OSM-side list at the same
GPS. The intersection is the "VLM-agreed" cohort (1,219 frames).

```bash
python -m src.a2_step1_gps_geo               # GPS → nearby attractions list
python -m src.a2_step2_vlm_geo               # VLM → visible attractions list
python -m src.a2_step3_gps_vlm_geo           # intersect → matched cohort
python -m src.a2_join_3way                   # build the canonical join table
python -m src.a2_match_strict                # strict-name matching pass
python -m src.a2_match_intersection          # softer matching pass
python -m src.a2_proximity_tag               # tag each match by distance
```

Outputs: `data/cities/zurich/a2/{GPS_GEO,VLM_GEO,GPS_VLM_GEO,join_3way}.jsonl`.

### Stage 6 — Heading-v2 + target-frame + route computation

```bash
python -m src.a2_heading_v2                  # gap-tiered heading per frame
python -m src.a2_target_frames               # pick training-worthy frames
python -m src.a2_destination_targets         # build (frame, destination) pairs
python -m src.a2_route                       # OSM-route → first-bearing → GT verb
python -m src.a2_sv_pano_attractions         # (helper) attractions per SV pano
```

Outputs: `data/cities/zurich/a2/heading_v2.jsonl` (15,053 rows),
`data/cities/zurich/a2/routes.jsonl` (3,657 (frame, destination) pairs
with route polyline + GT verb).

### Stage 7 — Teacher annotation (Gemini Pro 2.5)

Send each (frame, destination) pair to Gemini with the "Zurich
walking-tour guide" system prompt + variant-A user prompt (heading
given). Saves `<thinking>` + `<answer>` + format/direction passes.

```bash
python -m src.a2_annotate --limit 0          # full run, ~3 h via 3 parallel keys
python -m src.a2_sanity_check                # verify every required field is present
```

Cost: ~$25 in Gemini API spend. Output:
`data/cities/zurich/a2/annotations_a2_base.jsonl` (3,657 rows).

### Stage 8 — SFT split (V-stripped, `_nov` namespace)

Convert the base annotations into Qwen-chat SFT JSONL, with the
"Visible landmarks at this spot:" block REMOVED from the student
prompt (honor-code-clean: the student never sees the cheat-list
that helped the teacher). Produces 3 variants × {train, val, test}
= 9 files.

```bash
python -m src.a2_to_sft --variant given    --strip-visible --suffix _nov --only-pass
python -m src.a2_to_sft --variant derived  --strip-visible --suffix _nov --only-pass
python -m src.a2_to_sft --variant implicit --strip-visible --suffix _nov --only-pass
```

Outputs: `data/sft/a2_{given,derived,implicit}_{train,val,test}_nov.jsonl`.

### Stage 9 — LoRA SFT on Modal A100-80GB

```bash
# Per (variant, rank) — 9 trainings total. Each ~30 min wall-time.
for variant in given derived implicit; do
  for rank in 4 8 16; do
    modal run -d src.a2_train_modal::main \
        --variant $variant --lora-r $rank \
        --total-epochs 5 --suffix _nov
  done
done
```

Cost: ~$60 in Modal credits for all 9 trainings. Adapters land in
the `navlm-ckpts` volume at `/ckpts/lora_a2_<v>_r<R>_e5_nov/`.

### Stage 10 — Eval + scoring

```bash
# Zero-shot baselines (3 variants)
for variant in given derived implicit; do
  modal run src.a2_eval_modal::main --variant $variant --suffix _nov
done

# Trained-eval at e3 (intermediate checkpoint) and e5 (final adapter)
# 9 adapters × {e3, e5} = 18 trained-evals
for variant in given derived implicit; do
  for rank in 4 8 16; do
    for ckpt in 798 1340; do      # use the variant-specific step counts
      modal run src.a2_eval_modal::main \
          --variant $variant --suffix _nov \
          --adapter "/ckpts/lora_a2_${variant}_r${rank}_e5_nov/_trainer/checkpoint-${ckpt}"
    done
  done
done

# Pull results from Modal volume locally
modal volume get navlm-eval nov_<run_id>/ eval_pull/nov_<run_id>/ --force

# Local scoring (PASS rate per condition + per-row scoring)
python -m src.a2_score eval_pull/nov_<run_id>/
```

Outputs: `eval_pull/nov_<run_id>/<condition>/per_sample.jsonl` per
condition (21 in total) — each row tagged with `PASS`,
`format_pass`, `direction_pass`, `first_verb`, `gt_verb`, plus the
full `model_response` for qualitative analysis.

### (Optional) Paper figures + interactive viewers

```bash
python -m src.a2_figures_nov                 # fig 1, 2, 3, 3b, 3c
python -m src.a2_fig_map_vlm                 # fig 5 (success/failure narrative)
python -m src.a2_fig_match_examples          # fig 4 (DINOv2 match QC)
python -m src.a2_viz_matched                 # 2 HTML viewers: GPS map + VLM-agreed
python -m src.a2_viz_sft --strip-visible --suffix _nov   # SFT-prompt QC viewer
python -m src.a2_viz_route_gt                # route-GT illustration
```

## Reproducing the headline numbers

After completing stages 1–10:

```bash
python -m src.a2_score eval_pull/nov_<run_id>/
```

Expected output (best per variant):

| variant   | zs    | best LoRA     | absolute lift |
|-----------|------:|--------------:|--------------:|
| given     | 49.7 %| 98.8 % (r=8 e=5)  | +49.1 pp |
| derived   | 30.2 %| 67.2 % (r=16 e=3) | +37.0 pp |
| implicit  | 25.1 %| 62.2 % (r=16 e=5) | +37.1 pp |

## AI-tool-usage disclosure

Code in this submission was iteratively developed via Claude Code
(Anthropic Opus 4.7 / 4.8). A full session transcript is available
in the supplementary bundle at
`docs/supplementary/transcript_through_20260604_1921PDT.jsonl.gz`
 
 
## Contact

para2046 — `para2046 [at] stanford [dot] edu`
