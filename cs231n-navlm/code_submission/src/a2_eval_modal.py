"""Modal inference harness for Attempt 2 — 6 conditions.

The student model is **Qwen2.5-VL-7B**. Conditions differ in whether a
LoRA adapter is loaded and which input prompt template is used:

  zs-heading-given       : base Qwen + prompt-A (heading given)
  zs-heading-derived     : base Qwen + prompt-B (heading hidden; derive)
  zs-heading-implicit    : base Qwen + prompt-C (heading hidden; visual)
  trained-heading-given  : Qwen + lora_a2_given     + prompt-A
  trained-heading-derived: Qwen + lora_a2_derived   + prompt-B
  trained-heading-implicit: Qwen + lora_a2_implicit + prompt-C

Reads the held-out test split from the navlm-data volume:
   /sft/a2_{variant}_test.jsonl    (per condition's matching variant)

Note: zs-* conditions still read the corresponding variant's test
file because the user-prompt template needs to match (the test files
already have the right student_prompt baked into messages[user]).

Writes per-sample model outputs to navlm-eval at:
   /eval/<run_id>/<condition>/per_sample.jsonl

Scoring is done LOCALLY by `src/a2_score.py` (pull eval files, score
format/direction/anchor/interactive metrics).

  modal run src/a2_eval_modal.py --condition zs-heading-given
  modal run src/a2_eval_modal.py --condition trained-heading-derived \\
        --adapter /lora_a2_derived_r16_e2
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import modal

app = modal.App("navlm-eval-a2")

eval_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1", "torchvision",
        "transformers>=4.49", "peft>=0.13",
        "bitsandbytes>=0.44", "accelerate>=1.0",
        "qwen-vl-utils", "huggingface_hub", "pillow",
    )
)

ckpts = modal.Volume.from_name("navlm-ckpts", create_if_missing=True)
data_vol = modal.Volume.from_name("navlm-data", create_if_missing=True)
eval_vol = modal.Volume.from_name("navlm-eval", create_if_missing=True)

BASE_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
FRAMES_ROOT = "/data/frames"

CONDITION_TO_VARIANT = {
    "zs-heading-given":         "given",
    "zs-heading-derived":       "derived",
    "zs-heading-implicit":      "implicit",
    "trained-heading-given":    "given",
    "trained-heading-derived":  "derived",
    "trained-heading-implicit": "implicit",
}

DEFAULT_ADAPTER = {
    "trained-heading-given":    "/ckpts/lora_a2_given_r16_e2",
    "trained-heading-derived":  "/ckpts/lora_a2_derived_r16_e2",
    "trained-heading-implicit": "/ckpts/lora_a2_implicit_r16_e2",
}


def _resolve_image(image_rel):
    p = Path(image_rel)
    if p.is_absolute() and p.exists():
        return str(p)
    return str(Path(FRAMES_ROOT) / image_rel)


@app.function(
    image=eval_image,
    gpu="A100-40GB",
    timeout=3 * 3600,
    volumes={"/ckpts": ckpts, "/data": data_vol, "/eval": eval_vol},
    secrets=[modal.Secret.from_name("huggingface")],
)
def evaluate_condition(condition: str,
                        run_id: str,
                        adapter: str = "",
                        max_new_tokens: int = 4096,
                        temperature: float = 0.0,
                        limit: int = 0,
                        suffix: str = "") -> dict:
    """Run inference for one condition on its test split."""
    import time
    import torch
    from PIL import Image
    from peft import PeftModel
    from transformers import (AutoProcessor, BitsAndBytesConfig,
                              Qwen2_5_VLForConditionalGeneration)

    assert condition in CONDITION_TO_VARIANT, condition
    variant = CONDITION_TO_VARIANT[condition]
    is_trained = condition.startswith("trained-")

    if is_trained and not adapter:
        adapter = DEFAULT_ADAPTER[condition]

    test_path = Path(f"/data/sft/a2_{variant}_test{suffix}.jsonl")
    assert test_path.exists(), (
        f"{test_path} not on navlm-data — upload it first")
    rows = [json.loads(l) for l in test_path.open(encoding="utf-8")
            if l.strip()]
    if limit:
        rows = rows[:limit]
    print(f"[eval.{condition}] {len(rows)} test samples "
          f"(variant={variant}, trained={is_trained})", flush=True)

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    processor = AutoProcessor.from_pretrained(BASE_MODEL,
                                              max_pixels=448 * 448)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        BASE_MODEL, quantization_config=bnb,
        torch_dtype=torch.bfloat16, device_map="auto",
    )
    if is_trained:
        print(f"[eval.{condition}] loading adapter from {adapter}",
              flush=True)
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()

    # When an adapter is loaded, include its rank/epoch suffix in the
    # output dir so multi-rank sweeps don't overwrite each other.
    # We look at the FULL adapter path so that intermediate checkpoints
    # like /ckpts/lora_a2_given_r4_e5_nov/_trainer/checkpoint-963 also
    # get a unique suffix that includes the rank.
    import re as _re
    suffix = ""
    if is_trained and adapter:
        # 1. rank+epoch from the parent adapter dir name (e.g. "_r4_e5")
        rk = ""
        rk_match = _re.search(r"_r\d+_e\d+", adapter)
        if rk_match:
            rk = rk_match.group(0)
        # 2. checkpoint number from the leaf (when intermediate)
        ck = ""
        ck_match = _re.search(r"checkpoint-(\d+)", adapter)
        if ck_match:
            ck = "_ckpt" + ck_match.group(1)
        suffix = rk + ck
        if not suffix:
            suffix = "_" + Path(adapter).name
    out_dir = Path(f"/eval/{run_id}/{condition}{suffix}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "per_sample.jsonl"

    n_done = 0
    t0 = time.time()
    with out_path.open("w", encoding="utf-8") as fout:
        for i, row in enumerate(rows):
            img_path = _resolve_image(row["image_rel"])
            try:
                img = Image.open(img_path).convert("RGB")
            except (FileNotFoundError, OSError) as e:
                print(f"  skip {img_path}: {e}", flush=True)
                continue

            # build chat input (DROP the assistant turn — model generates it)
            messages_for_inference = [m for m in row["messages"]
                                       if m["role"] != "assistant"]
            text = processor.apply_chat_template(
                messages_for_inference, tokenize=False,
                add_generation_prompt=True)
            inputs = processor(text=[text], images=[img], padding=True,
                                return_tensors="pt").to(model.device)

            with torch.no_grad():
                gen = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False if temperature == 0.0 else True,
                    temperature=(temperature if temperature > 0 else 1.0),
                )
            # decode only the newly-generated portion
            gen_ids = gen[0][inputs["input_ids"].shape[1]:]
            response = processor.tokenizer.decode(
                gen_ids, skip_special_tokens=True)

            out_row = {
                "video": row["video"], "frame_id": row["frame_id"],
                "destination": row["destination"],
                "destination_zh": row.get("destination_zh", ""),
                "condition": condition, "variant": variant,
                "is_trained": is_trained,
                "adapter": adapter if is_trained else "",
                "image_rel": row["image_rel"],
                "gt_verb": row.get("gt_verb"),
                "teacher_first_verb": row.get("first_verb"),
                "teacher_direction_pass": row.get("direction_pass"),
                "heading": row.get("heading"),
                "model_response": response,
            }
            fout.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            fout.flush()
            n_done += 1
            if (i + 1) % 25 == 0 or (i + 1) == len(rows):
                elapsed = time.time() - t0
                eta = (len(rows) - i - 1) / max(1e-3, n_done / elapsed)
                print(f"  [{i+1:4d}/{len(rows)}] elapsed {elapsed/60:.1f}m "
                      f"ETA {eta/60:.1f}m", flush=True)

    summary = {"condition": condition, "variant": variant,
               "is_trained": is_trained, "adapter": adapter,
               "n_samples": n_done,
               "wall_time_s": time.time() - t0,
               "out_path": str(out_path)}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    eval_vol.commit()
    print(f"[eval.{condition}] DONE — {n_done} samples in "
          f"{(time.time()-t0)/60:.1f} min → {out_path}", flush=True)
    return summary


@app.local_entrypoint()
def main(condition: str = "zs-heading-given",
         run_id: str = "",
         adapter: str = "",
         max_new_tokens: int = 4096,
         limit: int = 0,
         suffix: str = ""):
    import datetime as dt
    if not run_id:
        run_id = dt.datetime.now().strftime("%Y%m%d_%H%M%S") + "_a2"
    result = evaluate_condition.remote(
        condition=condition, run_id=run_id, adapter=adapter,
        max_new_tokens=max_new_tokens, limit=limit, suffix=suffix)
    print("=== EVAL DONE ===")
    print(json.dumps(result, indent=2))
    print(f"\nPull results:  modal volume get navlm-eval "
          f"{run_id}/{condition} ./")
