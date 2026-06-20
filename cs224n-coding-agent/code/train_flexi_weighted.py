#!/usr/bin/env python3
import unsloth  # must be first (Unsloth patches before other imports)
"""
Qwen3-Coder-30B-A3B SFT training for Colab H100.
Supports 80k max_seq_length with Unsloth + QLoRA.

**Supervises only on agent (assistant) tokens** - full trajectory, loss only on assistant parts.
"""
import os
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import argparse
import math
import re
from pathlib import Path

import torch
from torch.utils.data import Dataset as TorchDataset
from transformers import Trainer, TrainingArguments
import torch.nn.functional as F

try:
    from unsloth import FastLanguageModel
except ImportError:
    raise ImportError(
        "Install unsloth first: pip install 'unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git'"
    )


def _extract_content(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list) and content:
        part = content[0]
        if isinstance(part, dict) and "text" in part:
            return part["text"]
        if isinstance(part, str):
            return part
    return ""


def _normalize_message(msg: dict) -> dict[str, str] | None:
    role = str(msg.get("role", "")).strip()
    if role not in {"system", "user", "assistant"}:
        return None
    content = _extract_content(msg.get("content"))
    if content == "":
        return None
    return {"role": role, "content": content}


def _assistant_char_spans(text: str) -> list[tuple[int, int]]:
    pattern = r"<\|im_start\|>assistant\n(.*?)<\|im_end\|>"
    spans = []
    for m in re.finditer(pattern, text, re.DOTALL):
        spans.append((m.start(1), m.end(1)))
    return spans


def _is_in_any_span(char_start: int, char_end: int, spans: list[tuple[int, int]]) -> bool:
    for s, e in spans:
        if char_start < e and char_end > s:
            return True
    return False


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="sft_train.jsonl")
    p.add_argument("--eval_data", default=None)
    p.add_argument("--eval_steps", type=int, default=20)
    p.add_argument("--model_name", default="Qwen/Qwen3-Coder-30B-A3B-Instruct")
    p.add_argument("--output_dir", default="./qwen_sft_output")
    p.add_argument("--max_seq_length", type=int, default=8192)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--window_turns", type=int, default=0)
    p.add_argument("--window_stride", type=int, default=0)
    p.add_argument("--keep_short_trajectory", action="store_true", default=True)
    p.add_argument("--drop_short_trajectory", action="store_false", dest="keep_short_trajectory")
    p.add_argument("--save_steps", type=int, default=0)
    p.add_argument("--loss_reduction", choices=["token_mean", "sample_mean"], default="sample_mean")
    p.add_argument("--print_gt_samples", type=int, default=2)
    p.add_argument("--use_phase1_weights", action="store_true", help="Use per-sample loss weights computed from meta.phase1 annotations.")
    p.add_argument("--phase1_weight_scale", type=float, default=0.8)
    p.add_argument("--phase1_with_check_bonus", type=float, default=0.4)
    p.add_argument("--phase1_no_check_penalty", type=float, default=0.6)
    p.add_argument("--phase1_weight_min", type=float, default=0.3)
    p.add_argument("--phase1_weight_max", type=float, default=3.0)
    p.add_argument("--use_sequence_aux_loss", action="store_false", help="Enable sequence-level auxiliary loss for modify->check behavior.")
    p.add_argument("--sequence_aux_lambda", type=float, default=0.1)
    p.add_argument("--sequence_good_threshold", type=float, default=0.5)
    return p.parse_args()


class AgentOnlySFTDataset(TorchDataset):
    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_seq_length: int,
        max_samples: int | None = None,
        window_turns: int = 0,
        window_stride: int = 0,
        keep_short_trajectory: bool = False,
        use_phase1_weights: bool = False,
        phase1_weight_scale: float = 0.8,
        phase1_with_check_bonus: float = 0.4,
        phase1_no_check_penalty: float = 0.6,
        phase1_weight_min: float = 0.3,
        phase1_weight_max: float = 3.0,
        sequence_good_threshold: float = 0.5,
    ):
        path = Path(data_path)
        if path.suffix == ".jsonl" and path.exists():
            from datasets import load_dataset as hf_load
            ds = hf_load("json", data_files=str(path), split="train")
        elif "/" in data_path and not Path(data_path).exists():
            from datasets import load_dataset as hf_load
            ds = hf_load(data_path, split="train")
        else:
            raise FileNotFoundError(f"Data not found: {data_path}")

        self.examples = []
        self.window_turns = max(0, int(window_turns))
        self.window_stride = max(0, int(window_stride))
        self.keep_short_trajectory = bool(keep_short_trajectory)
        self.use_phase1_weights = bool(use_phase1_weights)
        self.phase1_weight_scale = float(phase1_weight_scale)
        self.phase1_with_check_bonus = float(phase1_with_check_bonus)
        self.phase1_no_check_penalty = float(phase1_no_check_penalty)
        self.phase1_weight_min = float(phase1_weight_min)
        self.phase1_weight_max = float(phase1_weight_max)
        self.sequence_good_threshold = float(sequence_good_threshold)
        max_examples = max_samples if (max_samples and max_samples > 0) else None

        for ex in ds:
            sample_weight = self._compute_sample_weight(ex)
            sequence_label, sequence_mask = self._compute_sequence_target(ex)
            for clean in self._expand_example(ex):
                text = tokenizer.apply_chat_template(clean, tokenize=False, add_generation_prompt=False)
                self.examples.append(
                    {
                        "text": text,
                        "sample_weight": sample_weight,
                        "sequence_label": sequence_label,
                        "sequence_mask": sequence_mask,
                    }
                )
                if max_examples is not None and len(self.examples) >= max_examples:
                    break
            if max_examples is not None and len(self.examples) >= max_examples:
                break

        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def _compute_sample_weight(self, ex: dict) -> float:
        if not self.use_phase1_weights:
            return 1.0
        try:
            phase1 = ex.get("meta", {}).get("phase1", {})
            if not isinstance(phase1, dict):
                return 1.0
            score = float(phase1.get("sequence_quality_score", 0.5))
            score = max(0.0, min(1.0, score))
            modify_count = int(phase1.get("modify_count", 0) or 0)
            mtc = int(phase1.get("modify_then_check_count", 0) or 0)
            mwc = int(phase1.get("modify_without_check_count", 0) or 0)
            weight = 1.0 + self.phase1_weight_scale * ((score - 0.5) * 2.0)
            if modify_count > 0:
                mtc_ratio = max(0.0, min(1.0, mtc / max(1, modify_count)))
                mwc_ratio = max(0.0, min(1.0, mwc / max(1, modify_count)))
                weight += self.phase1_with_check_bonus * mtc_ratio
                weight -= self.phase1_no_check_penalty * mwc_ratio
            return float(max(self.phase1_weight_min, min(self.phase1_weight_max, weight)))
        except Exception:
            return 1.0

    def _compute_sequence_target(self, ex: dict) -> tuple[float, float]:
        try:
            phase1 = ex.get("meta", {}).get("phase1", {})
            if not isinstance(phase1, dict):
                return 0.0, 0.0
            modify_count = int(phase1.get("modify_count", 0) or 0)
            mtc = int(phase1.get("modify_then_check_count", 0) or 0)
            if modify_count <= 0:
                return 0.0, 0.0
            ratio = mtc / max(1, modify_count)
            label = 1.0 if ratio >= self.sequence_good_threshold else 0.0
            return label, 1.0
        except Exception:
            return 0.0, 0.0

    def _expand_example(self, ex: dict) -> list[list[dict[str, str]]]:
        if "messages" in ex and isinstance(ex.get("messages"), list):
            clean = []
            for m in ex.get("messages", []):
                if not isinstance(m, dict):
                    continue
                nm = _normalize_message(m)
                if nm is not None:
                    clean.append(nm)
            return [clean] if clean else []

        if "header" in ex and "turns" in ex:
            header = ex.get("header", {})
            turns = ex.get("turns", [])
            if not isinstance(header, dict) or not isinstance(turns, list):
                return []
            system_msg = _normalize_message(header.get("system")) if isinstance(header.get("system"), dict) else None
            user_msg = _normalize_message(header.get("user")) if isinstance(header.get("user"), dict) else None
            if system_msg is None or user_msg is None:
                return []

            clean_turns: list[dict[str, str]] = []
            for m in turns:
                if not isinstance(m, dict):
                    continue
                nm = _normalize_message(m)
                if nm is None:
                    continue
                if nm["role"] in {"assistant", "user"}:
                    clean_turns.append(nm)
            if not clean_turns:
                return []

            if self.window_turns <= 0:
                return [[system_msg, user_msg] + clean_turns]

            assistant_idx = [i for i, m in enumerate(clean_turns) if m["role"] == "assistant"]
            if len(assistant_idx) < self.window_turns:
                return [[system_msg, user_msg] + clean_turns] if self.keep_short_trajectory else []

            stride = self.window_stride if self.window_stride > 0 else self.window_turns
            samples: list[list[dict[str, str]]] = []
            for start in range(0, len(assistant_idx) - self.window_turns + 1, stride):
                first_a = assistant_idx[start]
                last_a = assistant_idx[start + self.window_turns - 1]
                body_start = first_a - 1 if first_a - 1 >= 0 and clean_turns[first_a - 1]["role"] == "user" else first_a
                body = clean_turns[body_start : last_a + 1]
                if not body or body[-1]["role"] != "assistant":
                    continue
                if sum(1 for m in body if m["role"] == "assistant") != self.window_turns:
                    continue
                sample = [system_msg, user_msg] + body
                if (
                    len(sample) >= 3
                    and sample[1]["role"] == "user"
                    and sample[2]["role"] == "user"
                    and sample[1]["content"] == sample[2]["content"]
                ):
                    sample = [sample[0], sample[1]] + sample[3:]
                samples.append(sample)
            return samples
        return []

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        text = self.examples[idx]["text"]
        enc = self.tokenizer(text, truncation=True, max_length=self.max_seq_length, padding=False, return_tensors=None, return_offsets_mapping=True)
        input_ids = enc["input_ids"]
        offset_mapping = enc.get("offset_mapping")
        if not offset_mapping:
            labels = input_ids.copy()
        else:
            spans = _assistant_char_spans(text)
            labels = []
            for i in range(len(input_ids)):
                if i + 1 < len(input_ids):
                    next_s, next_e = offset_mapping[i + 1]
                    labels.append(input_ids[i + 1] if spans and _is_in_any_span(next_s, next_e, spans) else -100)
                else:
                    labels.append(-100)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
            "sample_weight": torch.tensor(float(self.examples[idx].get("sample_weight", 1.0)), dtype=torch.float32),
            "sequence_label": torch.tensor(float(self.examples[idx].get("sequence_label", 0.0)), dtype=torch.float32),
            "sequence_mask": torch.tensor(float(self.examples[idx].get("sequence_mask", 0.0)), dtype=torch.float32),
        }


def agent_only_collate_fn(examples: list[dict], tokenizer) -> dict:
    max_len = max(e["input_ids"].size(0) for e in examples)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    input_ids = []
    labels = []
    attention_mask = []
    sample_weight = []
    sequence_label = []
    sequence_mask = []

    for ex in examples:
        seq_len = ex["input_ids"].size(0)
        pad_len = max_len - seq_len
        input_ids.append(ex["input_ids"].tolist() + [pad_id] * pad_len)
        labels.append(ex["labels"].tolist() + [-100] * pad_len)
        attention_mask.append(ex["attention_mask"].tolist() + [0] * pad_len)
        sample_weight.append(float(ex.get("sample_weight", 1.0)))
        sequence_label.append(float(ex.get("sequence_label", 0.0)))
        sequence_mask.append(float(ex.get("sequence_mask", 0.0)))

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "sample_weight": torch.tensor(sample_weight, dtype=torch.float32),
        "sequence_label": torch.tensor(sequence_label, dtype=torch.float32),
        "sequence_mask": torch.tensor(sequence_mask, dtype=torch.float32),
    }


class AgentOnlyTrainer(Trainer):
    def __init__(
        self,
        *args,
        loss_reduction: str = "sample_mean",
        use_sequence_aux_loss: bool = False,
        sequence_aux_lambda: float = 0.1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.loss_reduction = loss_reduction
        self.use_sequence_aux_loss = bool(use_sequence_aux_loss)
        self.sequence_aux_lambda = float(sequence_aux_lambda)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs["labels"]
        sample_weight = inputs.get("sample_weight")
        sequence_label = inputs.get("sequence_label")
        sequence_mask = inputs.get("sequence_mask")
        model_inputs = {k: v for k, v in inputs.items() if k not in {"labels", "sample_weight", "sequence_label", "sequence_mask"}}
        if self.use_sequence_aux_loss:
            model_inputs["output_hidden_states"] = True
        outputs = model(**model_inputs)
        logits = outputs.logits

        vocab_size = logits.size(-1)
        token_loss = F.cross_entropy(
            logits.view(-1, vocab_size),
            labels.view(-1),
            ignore_index=-100,
            reduction="none",
        ).view_as(labels)
        valid_mask = (labels != -100).to(token_loss.dtype)
        token_loss = token_loss * valid_mask
        if sample_weight is None:
            sample_weight = torch.ones(labels.size(0), device=labels.device, dtype=token_loss.dtype)
        else:
            sample_weight = sample_weight.to(device=labels.device, dtype=token_loss.dtype)

        if self.loss_reduction == "sample_mean":
            per_sample_tokens = valid_mask.sum(dim=1).clamp(min=1.0)
            per_sample_loss = token_loss.sum(dim=1) / per_sample_tokens
            sft_loss = (per_sample_loss * sample_weight).sum() / sample_weight.sum().clamp(min=1.0)
        else:
            token_weights = sample_weight.unsqueeze(1) * valid_mask
            sft_loss = (token_loss * sample_weight.unsqueeze(1)).sum() / token_weights.sum().clamp(min=1.0)

        loss = sft_loss
        if self.use_sequence_aux_loss:
            if sequence_label is None or sequence_mask is None:
                sequence_loss = torch.zeros((), device=labels.device, dtype=sft_loss.dtype)
            else:
                if not hasattr(model, "sequence_aux_head"):
                    hidden_size = getattr(model.config, "hidden_size", None)
                    if hidden_size is None:
                        raise ValueError("model.config.hidden_size is required for sequence aux loss.")
                    model.sequence_aux_head = torch.nn.Linear(hidden_size, 1).to(labels.device)

                last_hidden = outputs.hidden_states[-1]
                am = model_inputs["attention_mask"].to(last_hidden.dtype)
                denom = am.sum(dim=1, keepdim=True).clamp(min=1.0)
                pooled = (last_hidden * am.unsqueeze(-1)).sum(dim=1) / denom
                seq_logits = model.sequence_aux_head(pooled).squeeze(-1)
                sequence_label = sequence_label.to(device=seq_logits.device, dtype=seq_logits.dtype)
                sequence_mask = sequence_mask.to(device=seq_logits.device, dtype=seq_logits.dtype)
                bce = F.binary_cross_entropy_with_logits(seq_logits, sequence_label, reduction="none")
                sequence_loss = (bce * sequence_mask).sum() / sequence_mask.sum().clamp(min=1.0)

            loss = sft_loss + self.sequence_aux_lambda * sequence_loss

        return (loss, outputs) if return_outputs else loss


def main():
    args = parse_args()

    import shutil
    for cache_path in ["/workspace/tmp/unsloth_compiled_cache", "/tmp/unsloth_compiled_cache"]:
        cache_dir = Path(cache_path)
        if cache_dir.exists():
            shutil.rmtree(cache_dir, ignore_errors=True)
            print(f"Cleared unsloth cache: {cache_dir}")

    print("Loading model and tokenizer...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
        dtype=torch.bfloat16,
        attn_implementation="eager",
    )

    print("Adding LoRA...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_alpha=32,
        lora_dropout=0.05,
    )

    print("Loading dataset (full trajectory, supervise only on assistant)...")
    dataset = AgentOnlySFTDataset(
        args.data,
        tokenizer,
        max_seq_length=args.max_seq_length,
        max_samples=args.max_samples,
        window_turns=args.window_turns,
        window_stride=args.window_stride,
        keep_short_trajectory=args.keep_short_trajectory,
        use_phase1_weights=args.use_phase1_weights,
        phase1_weight_scale=args.phase1_weight_scale,
        phase1_with_check_bonus=args.phase1_with_check_bonus,
        phase1_no_check_penalty=args.phase1_no_check_penalty,
        phase1_weight_min=args.phase1_weight_min,
        phase1_weight_max=args.phase1_weight_max,
        sequence_good_threshold=args.sequence_good_threshold,
    )
    if args.window_turns > 0:
        stride = args.window_stride if args.window_stride > 0 else args.window_turns
        print(f"Dataset size: {len(dataset)} windows (window_turns={args.window_turns}, stride={stride})")
    else:
        print(f"Dataset size: {len(dataset)} trajectories")
    if args.use_phase1_weights and len(dataset) > 0:
        ws = [float(x.get("sample_weight", 1.0)) for x in dataset.examples]
        print(
            "Phase1 sample_weight stats: "
            f"min={min(ws):.3f}, p50={sorted(ws)[len(ws)//2]:.3f}, "
            f"mean={sum(ws)/len(ws):.3f}, max={max(ws):.3f}"
        )

    if args.print_gt_samples > 0:
        n_preview = min(args.print_gt_samples, len(dataset))
        print(f"\nPreviewing GT samples: {n_preview}")
        for i in range(n_preview):
            ex = dataset[i]
            input_ids = ex["input_ids"].tolist()
            labels = ex["labels"].tolist()
            supervised_ids = [tid for tid, lab in zip(input_ids, labels) if lab != -100]
            gt_text = tokenizer.decode(supervised_ids, skip_special_tokens=False)
            print(f"\n[GT sample {i}] total_tokens={len(input_ids)}, supervised_tokens={len(supervised_ids)}")
            print(gt_text[:2000] + ("..." if len(gt_text) > 2000 else ""))

    eval_dataset = None
    eval_data_path = args.eval_data
    if eval_data_path is None:
        train_path = Path(args.data)
        default_val = train_path.parent / "sft_val.jsonl"
        if default_val.exists():
            eval_data_path = str(default_val)
    if eval_data_path and Path(eval_data_path).exists():
        eval_dataset = AgentOnlySFTDataset(
            eval_data_path,
            tokenizer,
            max_seq_length=args.max_seq_length,
            max_samples=None,
            window_turns=args.window_turns,
            window_stride=args.window_stride,
            keep_short_trajectory=args.keep_short_trajectory,
            use_phase1_weights=args.use_phase1_weights,
            phase1_weight_scale=args.phase1_weight_scale,
            phase1_with_check_bonus=args.phase1_with_check_bonus,
            phase1_no_check_penalty=args.phase1_no_check_penalty,
            phase1_weight_min=args.phase1_weight_min,
            phase1_weight_max=args.phase1_weight_max,
            sequence_good_threshold=args.sequence_good_threshold,
        )
        print(f"Eval dataset: {len(eval_dataset)} trajectories (eval every {args.eval_steps} steps)")
    else:
        print("No eval data found; skipping validation.")

    from functools import partial
    collate_fn = partial(agent_only_collate_fn, tokenizer=tokenizer)

    steps_per_epoch = math.ceil(len(dataset) / max(1, args.batch_size))
    optimizer_steps_per_epoch = math.ceil(steps_per_epoch / max(1, args.gradient_accumulation_steps))
    total_steps = max(1, int(math.ceil(optimizer_steps_per_epoch * args.epochs)))
    auto_save_steps = max(1, total_steps // 5)
    save_steps = args.save_steps if args.save_steps > 0 else auto_save_steps
    print(f"Checkpoint schedule: total_steps={total_steps}, save_steps={save_steps}")

    training_kwargs = dict(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        bf16=True,
        save_strategy="steps",
        save_steps=save_steps,
        logging_steps=5,
        warmup_ratio=0.03,
        dataloader_num_workers=0,
        report_to=["tensorboard"],
        logging_dir=str(Path(args.output_dir) / "runs"),
    )
    if eval_dataset is not None:
        training_kwargs["eval_strategy"] = "steps"
        training_kwargs["eval_steps"] = args.eval_steps
    training_args = TrainingArguments(**training_kwargs)

    trainer = AgentOnlyTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
        loss_reduction=args.loss_reduction,
        use_sequence_aux_loss=args.use_sequence_aux_loss,
        sequence_aux_lambda=args.sequence_aux_lambda,
    )

    print("Training...")
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Done. Saved to {args.output_dir}")


if __name__ == "__main__":
    main()
