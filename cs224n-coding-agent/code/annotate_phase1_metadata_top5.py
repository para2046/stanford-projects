#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


MODEL_TO_FILE = {
    "claude-sonnet-4-5-20250929": "base_claudesonnet45_battlesnake_r15complete.jsonl",
    "gpt-5": "base_gpt5_battlesnake_r15complete.jsonl",
    "gpt-5-mini": "base_gpt5mini_battlesnake_r15complete.jsonl",
    "claude-sonnet-4-20250514": "base_claudesonnet4_battlesnake_r15complete.jsonl",
    "gemini-2.5-pro": "base_gemini25pro_battlesnake_r15complete.jsonl",
}


def extract_bash_commands(text: str) -> list[str]:
    if not text:
        return []
    blocks = re.findall(r"```bash\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    out: list[str] = []
    for block in blocks:
        cmd = block.strip()
        if cmd:
            out.append(cmd.lower())
    return out


def split_first_command(command_block: str) -> str:
    first = command_block.splitlines()[0].strip()
    return first


def is_submit(cmd: str) -> bool:
    return "complete_task_and_submit_final_output" in cmd


def is_check(cmd: str) -> bool:
    if "py_compile" in cmd:
        return True
    if "pytest" in cmd:
        return True
    if re.search(r"\bpython\d?\s+.*test_.*\.py\b", cmd):
        return True
    if re.search(r"\btest_.*\.py\b", cmd) and ("python" in cmd or "pytest" in cmd):
        return True
    if re.search(r"grep.*(def move|run_server|__main__|main\.py)", cmd):
        return True
    return False


def is_modify_main(cmd: str) -> bool:
    if re.search(r"(cat\s*>|cat\s*<<).*main\.py", cmd):
        return True
    if re.search(r"\bcp\s+.*\.py\s+main\.py\b", cmd):
        return True
    if re.search(r"\bsed\s+-i\b.*main\.py", cmd):
        return True
    return False


def is_modify_code_other(cmd: str) -> bool:
    if re.search(r"(cat\s*>|cat\s*<<).*(\.py)\b", cmd):
        if "main.py" in cmd:
            return False
        if "analyze" in cmd or "test_" in cmd:
            return False
        return True
    return False


def is_strategy_read(cmd: str) -> bool:
    if "/logs/rounds" in cmd:
        return True
    if "results.json" in cmd:
        return True
    if "sim_" in cmd and ".jsonl" in cmd:
        return True
    if "round_" in cmd and ".jsonl" in cmd:
        return True
    return False


def is_strategy_analysis(cmd: str) -> bool:
    if re.search(r"(cat\s*>|cat\s*<<).*?(analyze|find_losses|compare_rounds).*\.py", cmd):
        return True
    if re.search(r"\bpython\d?\s+.*(analyze|find_losses|compare_rounds).*\.py\b", cmd):
        return True
    return False


def classify_primary(cmd: str) -> str:
    if is_submit(cmd):
        return "SUBMIT"
    if is_check(cmd):
        return "CODE_CHECK"
    if is_modify_main(cmd):
        return "MODIFY_MAIN"
    if is_modify_code_other(cmd):
        return "MODIFY_CODE_OTHER"
    if is_strategy_read(cmd):
        return "STRATEGY_READ"
    if is_strategy_analysis(cmd):
        return "STRATEGY_ANALYSIS"
    if re.search(r"\b(cat|head|tail|sed -n)\b.*main\.py", cmd) and "cat >" not in cmd and "cat <<" not in cmd:
        return "SAFE_READ_MAIN"
    if cmd.startswith("ls ") or cmd.startswith("ls-") or "cat readme" in cmd or "cat docs/" in cmd:
        return "SAFE_READ_MISC"
    return "OTHER"


def annotate_turn_sequence(turns: list[dict[str, Any]]) -> tuple[Counter, dict[str, Any]]:
    assistant_cmds: list[str] = []
    for msg in turns:
        if msg.get("role") != "assistant":
            continue
        content = str(msg.get("content", ""))
        for block in extract_bash_commands(content):
            assistant_cmds.append(split_first_command(block))

    primary_tags = [classify_primary(cmd) for cmd in assistant_cmds]
    counts = Counter(primary_tags)

    modify_indices = [i for i, t in enumerate(primary_tags) if t in {"MODIFY_MAIN", "MODIFY_CODE_OTHER"}]
    check_indices = [i for i, t in enumerate(primary_tags) if t == "CODE_CHECK"]
    submit_indices = [i for i, t in enumerate(primary_tags) if t == "SUBMIT"]

    modify_then_check = 0
    modify_without_check = 0
    modify_with_submit_no_check = 0
    for mi in modify_indices:
        next_check = next((ci for ci in check_indices if ci > mi), None)
        next_submit = next((si for si in submit_indices if si > mi), None)
        if next_check is not None and (next_submit is None or next_check < next_submit):
            modify_then_check += 1
        else:
            modify_without_check += 1
            if next_submit is not None:
                modify_with_submit_no_check += 1

    submit_after_any_check = 0
    for si in submit_indices:
        if any(ci < si for ci in check_indices):
            submit_after_any_check += 1

    score = 0.5
    if modify_indices:
        score += 0.25 * (modify_then_check / len(modify_indices))
        score -= 0.30 * (modify_without_check / len(modify_indices))
    if submit_indices:
        score += 0.15 * (submit_after_any_check / len(submit_indices))
    score = max(0.0, min(1.0, score))

    seq = {
        "assistant_command_count": len(assistant_cmds),
        "step_tag_counts": dict(counts),
        "has_modify": bool(modify_indices),
        "has_check": bool(check_indices),
        "has_submit": bool(submit_indices),
        "modify_count": len(modify_indices),
        "check_count": len(check_indices),
        "submit_count": len(submit_indices),
        "modify_then_check_count": modify_then_check,
        "modify_without_check_count": modify_without_check,
        "modify_submit_no_check_count": modify_with_submit_no_check,
        "submit_after_any_check_count": submit_after_any_check,
        "sequence_quality_score": round(score, 4),
    }
    return counts, seq


def annotate_file(in_path: Path, out_path: Path) -> dict[str, Any]:
    n_rows = 0
    total_cmds = 0
    total_modify = 0
    total_check = 0
    total_submit = 0
    total_modify_then_check = 0
    total_modify_without_check = 0
    primary_counter: Counter = Counter()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with in_path.open("r", encoding="utf-8") as rf, out_path.open("w", encoding="utf-8") as wf:
        for line in rf:
            if not line.strip():
                continue
            row = json.loads(line)
            turns = row.get("turns", [])
            counts, seq = annotate_turn_sequence(turns if isinstance(turns, list) else [])

            row.setdefault("meta", {})
            row["meta"]["phase1"] = seq
            wf.write(json.dumps(row, ensure_ascii=False) + "\n")

            n_rows += 1
            total_cmds += seq["assistant_command_count"]
            total_modify += seq["modify_count"]
            total_check += seq["check_count"]
            total_submit += seq["submit_count"]
            total_modify_then_check += seq["modify_then_check_count"]
            total_modify_without_check += seq["modify_without_check_count"]
            primary_counter.update(counts)

    return {
        "rows": n_rows,
        "assistant_commands": total_cmds,
        "modify": total_modify,
        "check": total_check,
        "submit": total_submit,
        "modify_then_check": total_modify_then_check,
        "modify_without_check": total_modify_without_check,
        "primary_tag_counts": dict(primary_counter),
        "out": str(out_path),
    }


def main() -> int:
    data_dir = Path("/home/ning/CodeClash/qwen/data")
    for model, fname in MODEL_TO_FILE.items():
        in_path = data_dir / fname
        out_path = data_dir / fname.replace(".jsonl", "_phase1meta.jsonl")
        if not in_path.exists():
            print(f"[skip] missing {in_path}")
            continue
        summary = annotate_file(in_path, out_path)
        print(
            f"model={model} rows={summary['rows']} cmds={summary['assistant_commands']} "
            f"modify={summary['modify']} check={summary['check']} submit={summary['submit']} "
            f"modify_then_check={summary['modify_then_check']} "
            f"modify_without_check={summary['modify_without_check']} "
            f"out={summary['out']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
