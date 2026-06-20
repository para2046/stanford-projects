#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


THOUGHT_RE = re.compile(r"\bTHOUGHT\s*:", re.IGNORECASE)
BASH_BLOCK_RE = re.compile(r"```bash\s*\n(.*?)```", re.IGNORECASE | re.DOTALL)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Quick format test (5 samples) for HF uploaded model."
    )
    p.add_argument(
        "--data",
        required=True,
        help="JSONL data path. Supports messages schema and base {header, turns} schema.",
    )
    p.add_argument(
        "--model_name",
        required=True,
        help="HF model repo or local merged model path, e.g. z050209/test-merged",
    )
    p.add_argument("--max_samples", type=int, default=5, help="Trajectories to test.")
    p.add_argument(
        "--max_turns_per_sample",
        type=int,
        default=1,
        help="Assistant turns to test per trajectory (1 keeps it quick).",
    )
    p.add_argument("--max_new_tokens", type=int, default=384)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--print_outputs", action="store_true")
    p.add_argument("--no_progress", action="store_true")
    return p.parse_args()


def _extract_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list) and content:
        first = content[0]
        if isinstance(first, str):
            return first
        if isinstance(first, dict) and "text" in first and isinstance(first["text"], str):
            return first["text"]
    return ""


def _normalize_message(msg: dict[str, Any]) -> dict[str, str] | None:
    role = str(msg.get("role", "")).strip()
    if role not in {"system", "user", "assistant"}:
        return None
    content = _extract_content(msg.get("content"))
    if not content:
        return None
    return {"role": role, "content": content}


def to_messages(record: dict[str, Any]) -> list[dict[str, str]]:
    if "messages" in record and isinstance(record.get("messages"), list):
        out: list[dict[str, str]] = []
        for m in record["messages"]:
            if isinstance(m, dict):
                nm = _normalize_message(m)
                if nm is not None:
                    out.append(nm)
        return out

    if "header" in record and "turns" in record:
        header = record.get("header", {})
        turns = record.get("turns", [])
        if not isinstance(header, dict) or not isinstance(turns, list):
            return []
        out = []
        for k in ("system", "user"):
            m = header.get(k)
            if isinstance(m, dict):
                nm = _normalize_message(m)
                if nm is not None:
                    out.append(nm)
        for m in turns:
            if isinstance(m, dict):
                nm = _normalize_message(m)
                if nm is not None and nm["role"] in {"user", "assistant"}:
                    out.append(nm)
        return out

    return []


def has_thought(text: str) -> bool:
    return bool(THOUGHT_RE.search(text or ""))


def bash_block_count(text: str) -> int:
    return len(BASH_BLOCK_RE.findall(text or ""))


def format_ok(text: str) -> bool:
    return has_thought(text) and bash_block_count(text) == 1


def load_records(path: Path, max_samples: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
            if max_samples > 0 and len(out) >= max_samples:
                break
    return out


def assistant_turn_indices(messages: list[dict[str, str]]) -> list[int]:
    return [i for i, m in enumerate(messages) if m.get("role") == "assistant"]


def main() -> int:
    args = parse_args()
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Data not found: {data_path}")

    print("Loading model/tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        use_fast=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="auto",
    )
    model.eval()

    records = load_records(data_path, args.max_samples)
    print(f"Loaded records: {len(records)} from {data_path}")

    eval_plan: list[tuple[list[dict[str, str]], str]] = []
    for rec in records:
        messages = to_messages(rec)
        if not messages:
            continue
        turn_idxs = assistant_turn_indices(messages)
        if args.max_turns_per_sample > 0:
            turn_idxs = turn_idxs[: args.max_turns_per_sample]
        for idx in turn_idxs:
            context = messages[:idx]
            target = messages[idx]["content"]
            if context and target:
                eval_plan.append((context, target))

    print(f"Planned inference turns: {len(eval_plan)}")
    if not eval_plan:
        print("No assistant turns found.")
        return 0

    total = 0
    thought_ok = 0
    one_bash_ok = 0
    format_ok_count = 0

    iterator = eval_plan if args.no_progress else tqdm(eval_plan, desc="Testing", unit="turn")
    for i, (context, target) in enumerate(iterator, start=1):
        prompt = tokenizer.apply_chat_template(
            context,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.temperature > 0,
                temperature=max(args.temperature, 1e-5),
                pad_token_id=tokenizer.eos_token_id,
            )

        gen_ids = output_ids[0][inputs["input_ids"].shape[1] :]
        pred = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        total += 1
        h = has_thought(pred)
        b = bash_block_count(pred) == 1
        f = h and b
        thought_ok += int(h)
        one_bash_ok += int(b)
        format_ok_count += int(f)

        if args.print_outputs:
            print(f"\n===== SAMPLE TURN {i} =====")
            print(f"has_thought={h} one_bash={b} format_ok={f}")
            print("----- PRED -----")
            print(pred)
            print("----- TARGET (assistant ref) -----")
            print(target)

    print("\n--- Quick format verification ---")
    print(f"Total turns: {total}")
    print(f"Has THOUGHT: {thought_ok}/{total} ({100.0 * thought_ok / total:.1f}%)")
    print(f"Exactly 1 bash block: {one_bash_ok}/{total} ({100.0 * one_bash_ok / total:.1f}%)")
    print(f"Format OK (both): {format_ok_count}/{total} ({100.0 * format_ok_count / total:.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
