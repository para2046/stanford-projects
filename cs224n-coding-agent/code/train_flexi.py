#!/usr/bin/env python3
"""
Qwen3-Coder-30B-A3B SFT training for Colab H100.
Supports 80k max_seq_length with Unsloth + QLoRA.

**Supervises only on agent (assistant) tokens** - full trajectory, loss only on assistant parts.

Usage (Colab):
  !pip install -r requirements.txt
  !python train.py --data sft_train.jsonl --output_dir ./output
"""
from __future__ import annotations
import unsloth  # must be first

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
    p.add_argument("--data", default="sft_train.jsonl", help="Path to JSONL with 'messages' (list of {role, content}). Can be local path or HF dataset.")
    p.add_argument("--eval_data", default=None, help="Path to validation JSONL. If None, auto-detect sft_val.jsonl next to train data.")
    p.add_argument("--eval_steps", type=int, default=20, help="Run validation loss every N steps.")
    p.add_argument("--model_name", default="Qwen/Qwen3-Coder-30B-A3B-Instruct", help="Base model.")
    p.add_argument("--output_dir", default="./qwen_sft_output", help="Checkpoint output directory.")
    p.add_argument("--max_seq_length", type=int, default=8192, help="Max sequence length. 8192 for 24GB; 32768 for 48GB; 81920 for 80GB+.")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max_samples", type=int, default=None, help="Limit dataset size for debugging.")
    p.add_argument("--window_turns", type=int, default=0, help="Dynamic window size over assistant turns. 0 disables windowing (full trajectory). Used for base-format data with {header, turns}.")
    p.add_argument("--window_stride", type=int, default=0, help="Window stride over assistant turns when --window_turns > 0. 0 means use window_turns (no overlap).")
    p.add_argument("--keep_short_trajectory", action="store_true", default=True, help="When windowing is enabled and assistant turns are fewer than window_turns, keep the trajectory as one full sample instead of dropping it.")
    p.add_argument("--drop_short_trajectory", action="store_false", dest="keep_short_trajectory", help="Disable keeping short trajectories (drop when assistant turns < window_turns).")
    p.add_argument("--save_steps", type=int, default=0, help="Save checkpoint every N steps. 0 = auto (total_steps // 5).")
    p.add_argument("--loss_reduction", choices=["token_mean", "sample_mean"], default="sample_mean", help="How to aggregate assistant-only losses. token_mean = standard token-level mean over all supervised tokens; sample_mean = each sample contributes equally (recommended when lengths vary).")
    p.add_argument("--print_gt_samples", type=int, default=2, help="Print first N GT (assistant-supervised) samples with lengths after dataset build. Set 0 to disable.")
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
        max_examples = max_samples if (max_samples and max_samples > 0) else None

        for ex in ds:
            for clean in self._expand_example(ex):
                text = tokenizer.apply_chat_template(
                    clean,
                    tokenize=False,
                    add_generation_prompt=False,
                )
                self.examples.append({"text": text})
                if max_examples is not None and len(self.examples) >= max_examples:
                    break
            if max_examples is not None and len(self.examples) >= max_examples:
                break

        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

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

            system_msg = header.get("system")
            user_msg = header.get("user")
            if not isinstance(system_msg, dict) or not isinstance(user_msg, dict):
                return []
            system_msg = _normalize_message(system_msg)
            user_msg = _normalize_message(user_msg)
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
                if self.keep_short_trajectory:
                    return [[system_msg, user_msg] + clean_turns]
                return []

            stride = self.window_stride if self.window_stride > 0 else self.window_turns
            samples: list[list[dict[str, str]]] = []
            for start in range(0, len(assistant_idx) - self.window_turns + 1, stride):
                first_a = assistant_idx[start]
                last_a = assistant_idx[start + self.window_turns - 1]
                body_start = first_a
                if first_a - 1 >= 0 and clean_turns[first_a - 1]["role"] == "user":
                    body_start = first_a - 1
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
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_seq_length,
            padding=False,
            return_tensors=None,
            return_offsets_mapping=True,
        )
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
                    if spans and _is_in_any_span(next_s, next_e, spans):
                        labels.append(input_ids[i + 1])
                    else:
                        labels.append(-100)
                else:
                    labels.append(-100)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
        }


def agent_only_collate_fn(examples: list[dict], tokenizer) -> dict:
    max_len = max(e["input_ids"].size(0) for e in examples)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    input_ids = []
    labels = []
    attention_mask = []

    for ex in examples:
        seq_len = ex["input_ids"].size(0)
        pad_len = max_len - seq_len
        input_ids.append(ex["input_ids"].tolist() + [pad_id] * pad_len)
        labels.append(ex["labels"].tolist() + [-100] * pad_len)
        attention_mask.append(ex["attention_mask"].tolist() + [0] * pad_len)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
    }


class AgentOnlyTrainer(Trainer):
    def __init__(self, *args, loss_reduction: str = "sample_mean", **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_reduction = loss_reduction

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs["labels"]
        model_inputs = {k: v for k, v in inputs.items() if k != "labels"}
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

        if self.loss_reduction == "sample_mean":
            per_sample_tokens = valid_mask.sum(dim=1).clamp(min=1.0)
            per_sample_loss = token_loss.sum(dim=1) / per_sample_tokens
            loss = per_sample_loss.mean()
        else:
            loss = token_loss.sum() / valid_mask.sum().clamp(min=1.0)

        return (loss, outputs) if return_outputs else loss


def main():
    args = parse_args()

    print("Loading model and tokenizer...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
        dtype=torch.bfloat16,
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
    )
    if args.window_turns > 0:
        stride = args.window_stride if args.window_stride > 0 else args.window_turns
        print(f"Dataset size: {len(dataset)} windows (window_turns={args.window_turns}, stride={stride})")
    else:
        print(f"Dataset size: {len(dataset)} trajectories")

    if args.print_gt_samples > 0:
        n_preview = min(args.print_gt_samples, len(dataset))
        print(f"\nPreviewing GT samples: {n_preview}")
        for i in range(n_preview):
            ex = dataset[i]
            input_ids = ex["input_ids"].tolist()
            labels = ex["labels"].tolist()
            total_tokens = len(input_ids)
            supervised_ids = [tid for tid, lab in zip(input_ids, labels) if lab != -100]
            supervised_tokens = len(supervised_ids)
            gt_text = tokenizer.decode(supervised_ids, skip_special_tokens=False)
            print(
                f"\n[GT sample {i}] total_tokens={total_tokens}, "
                f"supervised_tokens={supervised_tokens}"
            )
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
    print(
        f"Checkpoint schedule: total_steps={total_steps}, "
        f"save_steps={save_steps} ({'manual' if args.save_steps > 0 else 'auto=total_steps//5'})"
    )

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
    )

    print("Training...")
    print(f"TensorBoard logs: {Path(args.output_dir) / 'runs'}. Run: tensorboard --logdir {args.output_dir}/runs")
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Done. Saved to {args.output_dir}")


if __name__ == "__main__":
    main()
