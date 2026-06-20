#!/usr/bin/env python3
"""
Annotate trajectories or dataset rows using GPT-5 or Claude API.
Produces [Obs][Thought][Act] THOUGHT format per annotate_trajectory_prompt.md.
Output: *_edited_gpt5.json / *_edited_claude.json (traj mode),
or *_edited_gpt5.jsonl / *_edited_claude.jsonl (jsonl mode).

Usage:
  # GPT-5 -> *_edited_gpt5.json
  OPENAI_API_KEY=sk-... uv run python qwen/annotate_phase2_claude_with_gpt5.py --model gpt-4o

  # Claude -> *_edited_claude.json
  ANTHROPIC_API_KEY=sk-... uv run python qwen/annotate_phase2_claude_with_gpt5.py --model claude-sonnet-4-5
  ANTHROPIC_API_KEY=sk-... uv run python qwen/annotate_phase2_claude_with_gpt5.py --jsonl-input /home/ning/CodeClash/qwen/data/base_claudesonnet45_battlesnake_r15complete.jsonl --model claude-sonnet-4-5 --limit 1
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def _ensure_openai() -> Any:
    try:
        from openai import OpenAI
        return OpenAI
    except ImportError:
        raise SystemExit("Install openai: pip install openai")


def _ensure_anthropic() -> Any:
    try:
        from anthropic import Anthropic
        return Anthropic
    except ImportError:
        raise SystemExit("Install anthropic: pip install anthropic")


PHASE2_ROOT = Path("/home/ning/CodeClash/logs/z0502/phase2")
BASH_BLOCK_RE = re.compile(r"```bash\s*\n(.*?)\n```", re.DOTALL)
OUTPUT_RE = re.compile(r"<output>\n?(.*?)\n?</output>", re.DOTALL)

SYSTEM_PROMPT = """You are annotating agent steps for a BattleSnake coding game. The agent edits a snake bot (main.py) across multiple rounds. For each assistant step, produce a structured THOUGHT with three parts: [Obs][Thought][Act].

- [Obs]: What we observed from previous output.
- [Thought]: Strategy and purpose-what is my approach, what is the goal. Be purposeful.
- [Act]: The actual action-concrete operation we're executing (explore/analyze/edit/test/submit).

Return ONLY the THOUGHT block (no bash block). No JSON, no extra explanation."""

USER_TEMPLATE = """## Task
Annotate this agent step. Produce a THOUGHT with exactly three sections.

## Context
- Game: BattleSnake. Agent improves main.py in /workspace. Logs in /logs/rounds/.
- Round: {round_num}
- Max steps: {max_steps}
- Current step index: {step_num} (1-based)

## Previous step
{prev_section}

## Current step (to annotate)
Original THOUGHT: {original_thought}
Command: {current_command}

## Output format
Output exactly:

THOUGHT: [Turn {step_num}/{max_steps}]
[Obs] (1-3 sentences) What we observed from the previous output. For step 1: "No previous output. Starting from game description." Key facts: scores, errors, file contents, state.
[Thought] (1-2 sentences) Strategy and purpose: what is my approach, what is the goal of this step. Be purposeful: e.g. "We need to avoid head-to-head when shorter. Goal: add length check before engaging."
[Act] (1 sentence) The actual action: concrete operation (explore/analyze/edit/test/submit) + brief description of what we're executing. E.g. "Edit main.py: add length check before head-to-head."

## Examples
Step 1: [Obs] No previous output. Starting from game description. [Thought] Need to understand repo structure first. Goal: list files before reading or editing. [Act] Explore: list repo with ls.
Step 3: [Obs] Round 1 lost 48-52. Died from head-to-head when shorter. main.py has no length check. [Thought] Strategy: avoid head-to-head when we're shorter. Purpose: add length check to reduce avoidable deaths. [Act] Edit main.py: add length check before head-to-head.

Return the THOUGHT block now."""


def extract_command(content: str) -> str | None:
    match = BASH_BLOCK_RE.search(content or "")
    if not match:
        return None
    return match.group(1).strip()


def extract_prev_output(user_content: str) -> str:
    match = OUTPUT_RE.search(user_content or "")
    if match:
        return match.group(1).strip()
    return (user_content or "").strip()


def call_openai(client: Any, model: str, system: str, user: str) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
    )
    return (resp.choices[0].message.content or "").strip()


def call_anthropic_http(api_key: str, model: str, system: str, user: str) -> str:
    url = "https://api.anthropic.com/v1/messages"
    body = {
        "model": model,
        "max_tokens": 1024,
        "system": system,
        "messages": [{"role": "user", "content": user}],
        "temperature": 0.2,
    }
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json; charset=utf-8",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        out = json.loads(resp.read().decode("utf-8"))
    blocks = out.get("content", [])
    text = blocks[0].get("text", "") if blocks else ""
    return text.strip()


def call_annotator(provider: str, client: Any, model: str, system: str, user: str) -> str:
    if provider == "anthropic":
        return call_anthropic_http(client, model, system, user)
    return call_openai(client, model, system, user)


def should_retry_api_error(error: Exception) -> bool:
    if isinstance(error, urllib.error.HTTPError) and error.code in (429, 503, 529):
        return True
    err = str(error).lower()
    return any(
        token in err
        for token in [
            "529",
            "overloaded",
            "rate limit",
            "timeout",
            "temporarily unavailable",
            "service unavailable",
        ]
    )


def call_annotator_with_retry(
    provider: str,
    client: Any,
    model: str,
    system: str,
    user: str,
    max_retries: int,
    retry_base_delay: float,
    source_name: str,
    turn_idx: int,
) -> str:
    attempt = 0
    while True:
        try:
            return call_annotator(provider, client, model, system, user)
        except Exception as e:
            if attempt >= max_retries or not should_retry_api_error(e):
                raise
            wait_s = retry_base_delay * (2**attempt)
            print(
                f"Retry {attempt + 1}/{max_retries} after API error "
                f"({source_name} step {turn_idx}): {e}. Waiting {wait_s:.1f}s...",
                flush=True,
            )
            time.sleep(wait_s)
            attempt += 1


def infer_round_num_from_name(name: str, default: int = 1) -> int:
    if "_r" not in name:
        return default
    try:
        return int(name.split("_r")[-1].split(".")[0])
    except ValueError:
        return default


def annotate_messages(
    messages: list[dict[str, Any]],
    source_name: str,
    client: Any,
    provider: str,
    model: str,
    round_num: int,
    dry_run: bool,
    delay_sec: float,
    prev_output_max_chars: int,
    original_thought_max_chars: int,
    max_retries: int,
    retry_base_delay: float,
) -> int:
    assistant_indices: list[int] = []
    for idx, msg in enumerate(messages):
        if msg.get("role") != "assistant":
            continue
        if extract_command(str(msg.get("content", ""))) is None:
            continue
        assistant_indices.append(idx)

    updated = 0
    for turn_idx, msg_idx in enumerate(assistant_indices, start=1):
        msg = messages[msg_idx]
        content = str(msg.get("content", ""))
        command = extract_command(content)
        if not command:
            continue

        original_thought = content
        if "THOUGHT:" in content:
            original_thought = content.split("THOUGHT:")[-1].split("```")[0].strip()
        if not original_thought:
            original_thought = content if content else "(no thought)"
        if original_thought_max_chars > 0 and len(original_thought) > original_thought_max_chars:
            original_thought = original_thought[:original_thought_max_chars] + "..."

        prev_command = ""
        prev_output = ""
        if turn_idx > 1:
            for j in range(msg_idx - 1, -1, -1):
                if messages[j].get("role") == "user":
                    prev_content = str(messages[j].get("content", ""))
                    prev_output = extract_prev_output(prev_content)
                    if prev_output_max_chars > 0 and len(prev_output) > prev_output_max_chars:
                        prev_output = prev_output[:prev_output_max_chars] + "..."
                    for k in range(j - 1, -1, -1):
                        if messages[k].get("role") == "assistant":
                            prev_command = extract_command(str(messages[k].get("content", ""))) or ""
                            break
                    break

        if turn_idx == 1:
            prev_section = "No previous output. This is the first step."
        else:
            prev_section = f"Command: {prev_command}\nOutput: {prev_output}"

        user_prompt = USER_TEMPLATE.format(
            round_num=round_num,
            max_steps=30,
            step_num=turn_idx,
            prev_section=prev_section,
            original_thought=original_thought,
            current_command=command,
        )

        if dry_run:
            print(f"[dry-run] {source_name} step {turn_idx}/30")
            continue

        print(f"[{source_name}] step {turn_idx}/30...", flush=True)
        try:
            enhanced = call_annotator_with_retry(
                provider=provider,
                client=client,
                model=model,
                system=SYSTEM_PROMPT,
                user=user_prompt,
                max_retries=max_retries,
                retry_base_delay=retry_base_delay,
                source_name=source_name,
                turn_idx=turn_idx,
            )
        except Exception as e:
            print(f"API error {source_name} step {turn_idx}: {e}")
            continue

        thought_only = (enhanced or "").strip()
        if thought_only:
            msg["content"] = f"{thought_only}\n\n```bash\n{command}\n```"
            updated += 1
        else:
            print(f"Unexpected response for {source_name} step {turn_idx}, skipping")

        if delay_sec > 0:
            time.sleep(delay_sec)

    return updated


def process_file(
    path: Path,
    client: Any,
    provider: str,
    model: str,
    round_num: int,
    dry_run: bool,
    delay_sec: float,
    output_suffix: str,
    prev_output_max_chars: int,
    original_thought_max_chars: int,
    max_retries: int,
    retry_base_delay: float,
) -> tuple[bool, int]:
    data = json.loads(path.read_text())
    messages = data.get("messages", [])
    updated = annotate_messages(
        messages=messages,
        source_name=path.name,
        client=client,
        provider=provider,
        model=model,
        round_num=round_num,
        dry_run=dry_run,
        delay_sec=delay_sec,
        prev_output_max_chars=prev_output_max_chars,
        original_thought_max_chars=original_thought_max_chars,
        max_retries=max_retries,
        retry_base_delay=retry_base_delay,
    )

    if not dry_run and updated > 0:
        out_path = path.with_name(path.name.replace(".traj.json", f"_edited_{output_suffix}.json"))
        out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2))

    return True, updated


def process_jsonl_file(
    input_path: Path,
    output_path: Path,
    client: Any,
    provider: str,
    model: str,
    dry_run: bool,
    delay_sec: float,
    limit: int,
    start_row: int,
    prev_output_max_chars: int,
    original_thought_max_chars: int,
    max_retries: int,
    retry_base_delay: float,
) -> tuple[int, int]:
    lines = input_path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return 0, 0

    start_idx = max(0, start_row - 1)
    selected_lines = lines[start_idx : start_idx + limit] if limit > 0 else lines[start_idx:]
    processed_rows = 0
    updated_turns = 0
    out_rows: list[str] = []

    for row_idx, line in enumerate(selected_lines, start=start_row):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            print(f"Skipping invalid JSON line {row_idx}")
            continue

        turns = row.get("turns", [])
        if not isinstance(turns, list):
            out_rows.append(line)
            continue

        meta = row.get("meta", {}) if isinstance(row.get("meta", {}), dict) else {}
        round_name = str(meta.get("round_name", ""))
        round_num = infer_round_num_from_name(round_name, default=1)

        source_name = f"{input_path.name}#row{row_idx}"
        updated = annotate_messages(
            messages=turns,
            source_name=source_name,
            client=client,
            provider=provider,
            model=model,
            round_num=round_num,
            dry_run=dry_run,
            delay_sec=delay_sec,
            prev_output_max_chars=prev_output_max_chars,
            original_thought_max_chars=original_thought_max_chars,
            max_retries=max_retries,
            retry_base_delay=retry_base_delay,
        )

        processed_rows += 1
        updated_turns += updated
        out_rows.append(json.dumps(row, ensure_ascii=False))

        if not dry_run and processed_rows > 0 and processed_rows % 10 == 0:
            output_path.write_text("\n".join(out_rows) + "\n", encoding="utf-8")
            print(f"Checkpoint: {processed_rows} rows saved to {output_path}", flush=True)

    if not dry_run and processed_rows > 0:
        output_path.write_text("\n".join(out_rows) + "\n", encoding="utf-8")

    return processed_rows, updated_turns


def main() -> None:
    ap = argparse.ArgumentParser(description="Annotate phase2 Claude trajectories via GPT-5 or Claude API")
    ap.add_argument("--root", type=Path, default=PHASE2_ROOT, help="Phase2 logs root")
    ap.add_argument("--jsonl-input", type=Path, default=None, help="Optional JSONL input to annotate (header/turns/meta rows)")
    ap.add_argument("--jsonl-output", type=Path, default=None, help="Optional JSONL output path (default: <input>_edited_<suffix>.jsonl)")
    ap.add_argument("--model", default="claude-sonnet-4-5", help="Model: gpt-4o, gpt-5 (OpenAI) or claude-sonnet-4-5 (Anthropic)")
    ap.add_argument("--provider", choices=["auto", "openai", "anthropic"], default="auto", help="API provider. auto = infer from model name")
    ap.add_argument("--dry-run", action="store_true", help="Only list steps, no API calls")
    ap.add_argument("--delay", type=float, default=1.0, help="Seconds between API calls")
    ap.add_argument("--output-suffix", default="auto", help="Output filename suffix: gpt5, claude, or auto (infer from provider)")
    ap.add_argument("--limit", type=int, default=0, help="Max number of rows/files to process (0 = all)")
    ap.add_argument("--start-row", type=int, default=1, help="JSONL only: 1-based row to start from (e.g. --start-row 1 --limit 150 = rows 1-150)")
    ap.add_argument("--prev-output-max-chars", type=int, default=0, help="Max chars for previous user output in prompt (0 = no truncation)")
    ap.add_argument("--original-thought-max-chars", type=int, default=1000, help="Max chars for original thought in prompt (0 = no truncation)")
    ap.add_argument("--max-retries", type=int, default=12, help="Retries for transient API errors like 529/overloaded")
    ap.add_argument("--retry-base-delay", type=float, default=3.0, help="Base delay seconds for exponential backoff retries")
    args = ap.parse_args()

    provider = args.provider
    if provider == "auto":
        provider = "anthropic" if args.model.lower().startswith("claude") else "openai"

    output_suffix = args.output_suffix
    if output_suffix == "auto":
        output_suffix = "claude" if provider == "anthropic" else "gpt5"

    client = None
    if not args.dry_run:
        if provider == "anthropic":
            key = os.environ.get("ANTHROPIC_API_KEY")
            if not key:
                raise SystemExit("Set ANTHROPIC_API_KEY for Claude API calls")
            client = key
        else:
            key = os.environ.get("OPENAI_API_KEY")
            if not key:
                raise SystemExit("Set OPENAI_API_KEY for OpenAI API calls")
            client = _ensure_openai()(api_key=key)

    if args.jsonl_input is not None:
        if not args.jsonl_input.exists():
            raise SystemExit(f"JSONL input not found: {args.jsonl_input}")
        output_path = args.jsonl_output
        if output_path is None:
            stem = args.jsonl_input.stem
            if args.start_row > 1 or args.limit > 0:
                end = args.start_row + args.limit - 1 if args.limit > 0 else "end"
                output_path = args.jsonl_input.with_name(
                    f"{stem}_edited_{output_suffix}_r{args.start_row}-{end}.jsonl"
                )
            else:
                output_path = args.jsonl_input.with_name(
                    f"{stem}_edited_{output_suffix}.jsonl"
                )
        print(f"JSONL mode: {args.jsonl_input}", flush=True)
        if args.start_row > 1 or args.limit > 0:
            end = args.start_row + args.limit - 1 if args.limit > 0 else "end"
            print(f"Processing rows {args.start_row} to {end}.", flush=True)
        processed_rows, updated_turns = process_jsonl_file(
            input_path=args.jsonl_input,
            output_path=output_path,
            client=client,
            provider=provider,
            model=args.model,
            dry_run=args.dry_run,
            delay_sec=args.delay,
            limit=args.limit,
            start_row=args.start_row,
            prev_output_max_chars=args.prev_output_max_chars,
            original_thought_max_chars=args.original_thought_max_chars,
            max_retries=args.max_retries,
            retry_base_delay=args.retry_base_delay,
        )
        print(f"Processed rows: {processed_rows}")
        print(f"Updated assistant turns: {updated_turns}")
        if processed_rows and not args.dry_run:
            print(f"Output: {output_path}")
        return

    if not args.root.exists():
        raise SystemExit(f"Root not found: {args.root}")

    traj_files = sorted(
        p
        for p in args.root.rglob("*.traj.json")
        if "/players/" in p.as_posix()
        and p.parent.name.startswith("claude-sonnet-4")
    )

    if not traj_files:
        raise SystemExit(f"No Claude trajectory files under {args.root}")

    if args.limit > 0:
        traj_files = traj_files[: args.limit]
        print(f"Limiting to {len(traj_files)} file(s).", flush=True)
    print(f"Found {len(traj_files)} trajectory files. Starting API calls...", flush=True)

    processed = 0
    updated_turns = 0
    for path in traj_files:
        round_num = infer_round_num_from_name(path.stem, default=1)

        ok, count = process_file(
            path,
            client=client,
            provider=provider,
            model=args.model,
            round_num=round_num,
            dry_run=args.dry_run,
            delay_sec=args.delay,
            output_suffix=output_suffix,
            prev_output_max_chars=args.prev_output_max_chars,
            original_thought_max_chars=args.original_thought_max_chars,
            max_retries=args.max_retries,
            retry_base_delay=args.retry_base_delay,
        )
        if ok:
            processed += 1
            updated_turns += count

    print(f"Processed files: {processed}")
    print(f"Updated assistant turns: {updated_turns}")
    if processed and not args.dry_run:
        print(f"Output: *_edited_{output_suffix}.json next to original files")


if __name__ == "__main__":
    main()
