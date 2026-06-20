#!/usr/bin/env bash
set -euo pipefail

# Clean example commands for the Qwen BattleSnake submission.
# Adjust paths as needed for your machine.

PROJECT_ROOT="/home/ning/CodeClash"
TRAIN_ROOT="/home/ning/CodeClash/backup/CodeClash"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"

export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/workspace/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/workspace/.cache/huggingface}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/workspace/.cache/xdg}"
export TMPDIR="${TMPDIR:-/workspace/tmp}"
export WANDB_DISABLED="${WANDB_DISABLED:-true}"
export TORCH_COMPILE_DISABLE="${TORCH_COMPILE_DISABLE:-1}"
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "Examples only. Uncomment the command you want to run."

# 1) Build the 8-agent base datasets
# cd "$PROJECT_ROOT"
# uv run python qwen/build_battlesnake_base_dataset_8agents_r15.py \
#   --root "$PROJECT_ROOT/codeclash_completed_full" \
#   --out-dir "$PROJECT_ROOT/qwen/data" \
#   --require-p2

# 2) Add phase-1 metadata for TQ-SFT
# cd "$PROJECT_ROOT"
# uv run python qwen/statistics/annotate_phase1_metadata_top5.py

# 3) ReAct annotation
# cd "$PROJECT_ROOT"
# export ANTHROPIC_API_KEY=YOUR_KEY
# uv run python qwen/annotate_phase2_claude_with_gpt5.py \
#   --jsonl-input "$PROJECT_ROOT/qwen/data/base_claudesonnet45_battlesnake_r15complete.jsonl" \
#   --model claude-sonnet-4-5 \
#   --start-row 1 \
#   --limit 430

# 4) Vanilla SFT
# cd "$TRAIN_ROOT/colab/qwen_sft"
# "$PYTHON_BIN" -u train_flexi.py \
#   --data "$PROJECT_ROOT/qwen/data/base_qwen_battlesnake.jsonl" \
#   --output_dir "$PROJECT_ROOT/outputs/vanilla_selfplay_sft" \
#   --window_turns 6 \
#   --window_stride 3 \
#   --max_seq_length 12000 \
#   --batch_size 1 \
#   --gradient_accumulation_steps 8 \
#   --epochs 1 \
#   --lr 1e-5 \
#   --loss_reduction sample_mean

# 5) ReAct SFT
# cd "$TRAIN_ROOT/colab/qwen_sft"
# "$PYTHON_BIN" -u train_flexi.py \
#   --data "$PROJECT_ROOT/qwen/data/base_claudesonnet45_battlesnake_r15complete_edited_claude_r1-430.jsonl" \
#   --output_dir "$PROJECT_ROOT/outputs/react_6turn" \
#   --window_turns 6 \
#   --window_stride 3 \
#   --max_seq_length 32000 \
#   --batch_size 1 \
#   --gradient_accumulation_steps 8 \
#   --epochs 2 \
#   --lr 1e-5 \
#   --loss_reduction sample_mean

# 6) TQ-SFT
# cd "$TRAIN_ROOT/colab/qwen_sft"
# "$PYTHON_BIN" -u train_flexi_weighted.py \
#   --data "$PROJECT_ROOT/qwen/data/base_top5_phase1meta_battlesnake_r15complete_merged.jsonl" \
#   --output_dir "$PROJECT_ROOT/outputs/tq_sft" \
#   --window_turns 6 \
#   --window_stride 6 \
#   --max_seq_length 12000 \
#   --batch_size 1 \
#   --gradient_accumulation_steps 8 \
#   --epochs 1 \
#   --lr 1e-5 \
#   --loss_reduction sample_mean \
#   --use_phase1_weights

# 7) Quick format test
# "$PYTHON_BIN" "$TRAIN_ROOT/qwen/test_flexi.py" \
#   --data "$PROJECT_ROOT/qwen/data/base_claudesonnet45_battlesnake_r15complete_edited_claude_r1-430.jsonl" \
#   --model_name /path/to/merged/model \
#   --max_samples 5 \
#   --max_turns_per_sample 1 \
#   --temperature 0.0 \
#   --print_outputs

# 8) Safe tournament evaluation
# cd "$PROJECT_ROOT"
# uv run python qwen/run_s1000_safe.py \
#   qwen/run_sft_6turns_vs_qwen30b_battlesnake_r15_s1000.yaml \
#   -s run_sft_6turns_vs_qwen30b_battlesnake_r15_s1000
