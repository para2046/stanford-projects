#!/usr/bin/env python3
"""Run CodeClash with patch for invalid sim JSON. Usage: uv run python qwen/run_s1000_safe.py qwen/run_qwen_hf_coder_temp.yaml"""
import json
import sys
from collections import defaultdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from codeclash.arenas.arena import RoundStats
from codeclash.agents.player import Player
from codeclash.constants import RESULT_TIE


def _apply_battlesnake_empty_scores_patch():
    import codeclash.arenas.battlesnake.battlesnake as bs_mod

    def _patched_get_results(self, agents: list[Player], round_num: int, stats: RoundStats) -> None:
        scores = defaultdict(int)
        available_players = [p.name for p in agents if p.name not in self._failed_to_start_player]
        if len(available_players) > 1:
            empty_files = []
            for idx in range(self.game_config["sims_per_round"]):
                sim_file = self.log_round(round_num) / f"sim_{idx}.jsonl"
                try:
                    if not sim_file.exists() or sim_file.stat().st_size == 0:
                        empty_files.append(idx)
                        continue
                    with open(sim_file) as f:
                        content = f.read().strip()
                        if not content:
                            empty_files.append(idx)
                            continue
                        lines = content.split("\n")
                        if not lines or not lines[-1]:
                            empty_files.append(idx)
                            continue
                        results = json.loads(lines[-1])
                        winner = RESULT_TIE if results["isDraw"] else results["winnerName"]
                        scores[winner] += 1
                except (FileNotFoundError, json.JSONDecodeError):
                    empty_files.append(idx)
            if empty_files:
                self.logger.warning(f"Round {round_num}: {len(empty_files)}/{self.game_config['sims_per_round']} sim files empty.")
            if not scores:
                sims = self.game_config["sims_per_round"]
                scores[RESULT_TIE] = sims
                for p in available_players:
                    scores[p] = 0
        else:
            if available_players:
                scores[available_players[0]] = self.game_config["sims_per_round"]
            else:
                scores[RESULT_TIE] = self.game_config["sims_per_round"]
        winner = max(scores, key=scores.get)
        winner = RESULT_TIE if list(scores.values()).count(scores[winner]) > 1 else winner
        stats.winner = winner
        stats.scores = dict(scores)
        for player, score in scores.items():
            if player != RESULT_TIE:
                stats.player_stats[player].score = score

    bs_mod.BattleSnakeArena.get_results = _patched_get_results


def main_cli() -> None:
    _apply_battlesnake_empty_scores_patch()
    import main as codeclash_main

    codeclash_main.main_cli()


if __name__ == "__main__":
    main_cli()
