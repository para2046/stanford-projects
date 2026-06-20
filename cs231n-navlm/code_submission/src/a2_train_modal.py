"""Modal LoRA fine-tuning of Qwen2.5-VL-7B for Attempt 2 — one of
three variants (given / derived / implicit).

Reads chat-template SFT data from the navlm-data volume at:
   /sft/a2_{variant}_train.jsonl
   /sft/a2_{variant}_val.jsonl

(produced by `src/a2_to_sft.py` and uploaded with
 `modal volume put navlm-data data/sft/a2_{variant}_*.jsonl /sft/`)

Saves the adapter to navlm-ckpts at:
   /lora_a2_{variant}_r16_e2/

  modal run src/a2_train_modal.py --variant given
  modal run src/a2_train_modal.py --variant derived --epochs 3
  modal run src/a2_train_modal.py --variant implicit --limit 32   # smoke
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import modal

app = modal.App("navlm-train-a2")

train_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1", "torchvision",
        "transformers>=4.49", "peft>=0.13",
        "bitsandbytes>=0.44", "accelerate>=1.0", "datasets",
        "qwen-vl-utils", "huggingface_hub", "pillow",
    )
)

ckpts = modal.Volume.from_name("navlm-ckpts", create_if_missing=True)
data_vol = modal.Volume.from_name("navlm-data", create_if_missing=True)

BASE_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]
FRAMES_ROOT = "/data/frames"


def _resolve_image(image_rel: str, frames_root: str) -> str:
    p = Path(image_rel)
    if p.is_absolute() and p.exists():
        return str(p)
    return str(Path(frames_root) / image_rel)


@app.function(
    image=train_image,
    gpu="A100-80GB",
    timeout=6 * 3600,
    volumes={"/ckpts": ckpts, "/data": data_vol},
    secrets=[modal.Secret.from_name("huggingface")],
)
def train_lora(variant: str = "given",
               epochs: int = 2,
               lr: float = 2e-4,
               lora_r: int = 16,
               lora_alpha: int = 32,
               limit: int = 0,
               resume_adapter: str = "",
               suffix: str = "") -> dict:
    """LoRA SFT for one Attempt-2 variant.

    If `resume_adapter` is set, load that saved LoRA adapter as the
    starting point (instead of a fresh LoRA). Optimizer + LR schedule
    are still fresh — cosine restarts from peak LR over the new total
    step count. The adapter's own r/alpha/dropout/target_modules
    override the CLI flags so we don't get a rank mismatch.
    """
    import re
    import torch
    from PIL import Image
    from peft import LoraConfig, PeftConfig, PeftModel, get_peft_model
    from transformers import (AutoProcessor, BitsAndBytesConfig,
                              EarlyStoppingCallback,
                              Qwen2_5_VLForConditionalGeneration,
                              Trainer, TrainingArguments)

    assert variant in ("given", "derived", "implicit"), variant
    train_path = Path(f"/data/sft/a2_{variant}_train{suffix}.jsonl")
    val_path = Path(f"/data/sft/a2_{variant}_val{suffix}.jsonl")
    assert train_path.exists(), (
        f"{train_path} not on navlm-data — upload first:\n"
        f"  modal volume put navlm-data data/sft/a2_{variant}_*.jsonl "
        f"/sft/")
    train_rows = [json.loads(l) for l in train_path.open(encoding="utf-8")
                   if l.strip()]
    val_rows = ([json.loads(l) for l in val_path.open(encoding="utf-8")
                  if l.strip()] if val_path.exists() else [])
    if limit:
        train_rows = train_rows[:limit]
        val_rows = val_rows[:max(1, limit // 8)]
    print(f"[train.a2.{variant}] train={len(train_rows)} val={len(val_rows)}"
          f" · {epochs} epochs · lr={lr} · r={lora_r}", flush=True)

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
    # ── Resume from existing adapter (Mode B: load weights, fresh optim) ──
    resume_orig_epochs = 0   # used for output-dir naming
    if resume_adapter:
        assert Path(resume_adapter).exists(), (
            f"resume_adapter not found: {resume_adapter}")
        # adapter config governs r/alpha/dropout/targets — CLI flags ignored
        cfg = PeftConfig.from_pretrained(resume_adapter)
        lora_r = cfg.r
        lora_alpha = cfg.lora_alpha
        model = PeftModel.from_pretrained(model, resume_adapter,
                                           is_trainable=True)
        # Parse "_e<N>" from adapter dir for output naming continuity
        m = re.search(r"_e(\d+)$", resume_adapter.rstrip("/"))
        if m:
            resume_orig_epochs = int(m.group(1))
        print(f"[train.a2.{variant}] RESUMED from {resume_adapter} "
              f"(r={lora_r}, alpha={lora_alpha}, orig_epochs="
              f"{resume_orig_epochs}); will train +{epochs} more epochs "
              f"with fresh cosine LR schedule", flush=True)
    else:
        model = get_peft_model(model, LoraConfig(
            r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.05,
            target_modules=LORA_TARGETS, task_type="CAUSAL_LM",
        ))
    model.print_trainable_parameters()

    # Pre-compute the assistant-turn marker so we only train on the model's
    # own output (<thinking>...</thinking><answer>...</answer>), NOT on the
    # repetitive system+user prompt tokens.
    asst_prefix_ids = processor.tokenizer.encode(
        "<|im_start|>assistant\n", add_special_tokens=False)
    apl = len(asst_prefix_ids)

    def collate(batch):
        texts, images = [], []
        for row in batch:
            img_path = _resolve_image(row["image_rel"], FRAMES_ROOT)
            try:
                img = Image.open(img_path).convert("RGB")
            except (FileNotFoundError, OSError) as e:
                print(f"  skip {img_path}: {e}", flush=True)
                continue
            text = processor.apply_chat_template(
                row["messages"], tokenize=False,
                add_generation_prompt=False)
            texts.append(text); images.append(img)
        if not texts:
            raise RuntimeError("batch contained no resolvable images")
        enc = processor(text=texts, images=images, padding=True,
                        return_tensors="pt")
        labels = enc["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100
        for tok in ("<|image_pad|>", "<|vision_start|>",
                    "<|vision_end|>"):
            tid = processor.tokenizer.convert_tokens_to_ids(tok)
            if isinstance(tid, int) and tid >= 0:
                labels[labels == tid] = -100
        # Mask system+user tokens: search for <|im_start|>assistant\n in
        # each row's input_ids and mask everything up to and including it.
        for i in range(labels.shape[0]):
            seq = enc["input_ids"][i].tolist()
            for j in range(len(seq) - apl + 1):
                if seq[j:j + apl] == asst_prefix_ids:
                    labels[i, :j + apl] = -100
                    break
        enc["labels"] = labels
        return enc

    # Output dir: if resumed, total_epochs = orig + new (so r4 from e3 + 2
    # more epochs writes to /ckpts/lora_a2_<v>_r4_e5).
    total_epochs = resume_orig_epochs + epochs
    out_dir = f"/ckpts/lora_a2_{variant}_r{lora_r}_e{total_epochs}{suffix}"
    args = TrainingArguments(
        output_dir=out_dir + "/_trainer", num_train_epochs=epochs,
        per_device_train_batch_size=1, gradient_accumulation_steps=8,
        learning_rate=lr, lr_scheduler_type="cosine", warmup_ratio=0.03,
        bf16=True,
        logging_steps=10, eval_strategy="epoch",
        per_device_eval_batch_size=1,
        # Save one checkpoint per epoch so EarlyStopping +
        # load_best_model_at_end can pick the epoch with the lowest
        # MASKED eval loss (covers only <thinking>+<answer> tokens).
        # save_total_limit=5: keep all 5 epoch checkpoints so we can
        # later eval at intermediate epochs (e.g. e3 vs e5 comparison
        # from a single 5-epoch run).
        save_strategy="epoch", save_total_limit=5,
        load_best_model_at_end=True, metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=[], remove_unused_columns=False,
    )
    trainer = Trainer(model=model, args=args,
                      train_dataset=train_rows, eval_dataset=val_rows,
                      data_collator=collate,
                      callbacks=[EarlyStoppingCallback(
                          early_stopping_patience=2)])
    trainer.train()
    val_metrics = trainer.evaluate()
    print(f"[train.a2.{variant}] final val: {val_metrics}", flush=True)

    model.save_pretrained(out_dir)
    hist = [{"step": s.get("step"), "loss": s.get("loss"),
             "eval_loss": s.get("eval_loss"), "epoch": s.get("epoch")}
            for s in trainer.state.log_history]
    (Path(out_dir) / "history.json").write_text(json.dumps(hist, indent=2))
    (Path(out_dir) / "summary.json").write_text(json.dumps({
        "variant": variant,
        "epochs_this_run": epochs,
        "resume_orig_epochs": resume_orig_epochs,
        "total_epochs": total_epochs,
        "resume_adapter": resume_adapter or None,
        "lr": lr,
        "lora_r": lora_r, "lora_alpha": lora_alpha,
        "n_train": len(train_rows), "n_val": len(val_rows),
        "final_eval_loss": val_metrics.get("eval_loss"),
    }, indent=2))
    ckpts.commit()
    print(f"[train.a2.{variant}] adapter saved -> {out_dir}", flush=True)
    return {"adapter_path": out_dir,
            "n_train": len(train_rows), "n_val": len(val_rows),
            "final_eval_loss": val_metrics.get("eval_loss"),
            "history": hist}


@app.local_entrypoint()
def main(variant: str = "given", epochs: int = 2, lr: float = 2e-4,
         lora_r: int = 16, lora_alpha: int = 0, limit: int = 0,
         resume_adapter: str = "", suffix: str = ""):
    """CLI wrapper.

    --resume-adapter <path>  : load a saved LoRA adapter (Mode B —
        fresh optimizer + fresh cosine LR over the new epoch count).
        e.g. --resume-adapter /ckpts/lora_a2_given_r4_e3
        Output will be /ckpts/lora_a2_<v>_r<r>_e<orig+new>/
        Adapter's own r/alpha override the CLI flags.
    """
    if lora_alpha == 0:
        lora_alpha = 2 * lora_r          # default alpha = 2 * rank
    result = train_lora.remote(variant=variant, epochs=epochs, lr=lr,
                               lora_r=lora_r, lora_alpha=lora_alpha,
                               limit=limit, resume_adapter=resume_adapter,
                               suffix=suffix)
    print("=== TRAIN DONE ===")
    print(json.dumps({k: v for k, v in result.items() if k != "history"},
                     indent=2))
    print(f"Pull adapter:  modal volume get navlm-ckpts "
          f"{result['adapter_path']} ./")
