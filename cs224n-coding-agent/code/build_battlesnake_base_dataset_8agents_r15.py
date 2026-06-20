#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


AGENT_OUTPUTS: list[tuple[str, str]] = [
    ("claude-sonnet-4-5-20250929", "base_claudesonnet45_battlesnake_r15complete.jsonl"),
    ("gpt-5-mini", "base_gpt5mini_battlesnake_r15complete.jsonl"),
    ("gpt-5", "base_gpt5_battlesnake_r15complete.jsonl"),
    ("grok-code-fast-1", "base_grokcodefast_battlesnake_r15complete.jsonl"),
    ("o3", "base_o3_battlesnake_r15complete.jsonl"),
    ("claude-sonnet-4-20250514", "base_claudesonnet4_battlesnake_r15complete.jsonl"),
    ("qwen3-coder-plus-2025-09-23", "base_qwen3coderplus_battlesnake_r15complete.jsonl"),
    ("gemini-2.5-pro", "base_gemini25pro_battlesnake_r15complete.jsonl"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build 8 BattleSnake base datasets with exact player matching and full r1~r15 coverage."
    )
    p.add_argument(
        "--root",
        default="/home/ning/CodeClash/codeclash_completed_full",
        help="Root directory containing tournament folders.",
    )
    p.add_argument(
        "--out-dir",
        default="/home/ning/CodeClash/qwen/data",
        help="Directory to write output JSONL files.",
    )
    p.add_argument(
        "--require-p2",
        action="store_true",
        help="Only include p2 tournaments.",
    )
    return p.parse_args()


def content_to_str(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out: list[str] = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                out.append(str(item["text"]))
            elif isinstance(item, str):
                out.append(item)
        return "\n".join(out)
    return str(content)


def normalize_message(msg: dict[str, Any]) -> dict[str, str]:
    return {
        "role": str(msg.get("role", "")),
        "content": content_to_str(msg.get("content", "")),
    }


def extract_header_and_turns(
    messages: list[dict[str, Any]],
) -> tuple[dict[str, str], dict[str, str], list[dict[str, str]]] | None:
    system_idx = None
    for i, m in enumerate(messages):
        if m.get("role") == "system":
            system_idx = i
            break
    if system_idx is None:
        return None

    first_user_idx = None
    for j in range(system_idx + 1, len(messages)):
        if messages[j].get("role") == "user":
            first_user_idx = j
            break
    if first_user_idx is None:
        return None

    header_system = normalize_message(messages[system_idx])
    header_user = normalize_message(messages[first_user_idx])

    turns: list[dict[str, str]] = []
    for m in messages[first_user_idx + 1 :]:
        role = m.get("role")
        if role in {"assistant", "user"}:
            turns.append(normalize_message(m))

    if not turns:
        return None
    return header_system, header_user, turns


def round_of_traj_file(path: Path) -> int | None:
    name = path.name
    if "_r" not in name or not name.endswith(".traj.json"):
        return None
    try:
        r_str = name.rsplit("_r", 1)[1].split(".traj.json", 1)[0]
        return int(r_str)
    except Exception:
        return None


def list_battlesnake_tournaments(root: Path, require_p2: bool) -> list[Path]:
    out: list[Path] = []
    for tdir in sorted(root.iterdir()):
        if not tdir.is_dir():
            continue
        name = tdir.name.lower()
        if "pvp" not in name or "battlesnake" not in name:
            continue
        if require_p2 and ".p2." not in name:
            continue
        if not (tdir / "players").is_dir():
            continue
        out.append(tdir)
    return out


def rows_for_agent(
    tournaments: list[Path],
    agent_name: str,
) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    kept_tournaments = 0
    seen_keys: set[tuple[str, str, str]] = set()

    required_rounds = set(range(1, 16))
    for tdir in tournaments:
        pdir = tdir / "players" / agent_name
        if not pdir.is_dir():
            continue

        traj_paths = sorted(pdir.glob("*.traj.json"))
        round_to_path: dict[int, Path] = {}
        for traj_path in traj_paths:
            r = round_of_traj_file(traj_path)
            if r is None:
                continue
            if r not in round_to_path:
                round_to_path[r] = traj_path

        if not required_rounds.issubset(set(round_to_path.keys())):
            continue

        kept_tournaments += 1
        for r in range(1, 16):
            traj_path = round_to_path[r]
            row_key = (str(tdir), agent_name, traj_path.name)
            if row_key in seen_keys:
                continue
            seen_keys.add(row_key)

            try:
                obj = json.loads(traj_path.read_text(encoding="utf-8"))
            except Exception:
                continue

            messages = obj.get("messages", [])
            if not isinstance(messages, list):
                continue

            parsed = extract_header_and_turns(messages)
            if parsed is None:
                continue
            header_system, header_user, turns = parsed
            rows.append(
                {
                    "header": {
                        "system": header_system,
                        "user": header_user,
                    },
                    "turns": turns,
                    "meta": {
                        "tournament_dir": str(tdir),
                        "tournament_folder_name": tdir.name,
                        "player_name": agent_name,
                        "traj_file": traj_path.name,
                        "round_name": f"r{r}",
                        "game_name": "BattleSnake",
                        "exit_status": obj.get("info", {}).get("exit_status", ""),
                    },
                }
            )

    return rows, kept_tournaments


def main() -> int:
    args = parse_args()
    root = Path(args.root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    tournaments = list_battlesnake_tournaments(root, require_p2=args.require_p2)
    print(f"battlesnake_tournaments_total={len(tournaments)}")

    for agent_name, out_name in AGENT_OUTPUTS:
        rows, kept_tournaments = rows_for_agent(tournaments, agent_name)
        out_path = out_dir / out_name
        with out_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        total_rounds = len(rows)
        print(
            f"agent={agent_name} tournaments={kept_tournaments} "
            f"rounds={total_rounds} out={out_path}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
