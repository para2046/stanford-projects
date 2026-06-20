# Qwen BattleSnake Submission Guide

This folder organizes the code and commands used for the CS224N report experiments on Qwen3-Coder-30B in CodeClash BattleSnake.

The Python files in this folder are direct copies of the experiment scripts we used, grouped into a single submission-oriented directory.

## Code Map

Submission scripts in this folder:

- `train_flexi.py`
- `train_flexi_weighted.py`
- `test_flexi.py`
- `build_battlesnake_base_dataset_8agents_r15.py`
- `annotate_phase1_metadata_top5.py`
- `annotate_phase2_claude_with_gpt5.py`
- `run_s1000_safe.py`
- `run_examples.sh`

- Vanilla SFT and ReAct SFT training:
  `/home/ning/CodeClash/backup/CodeClash/colab/qwen_sft/train_flexi.py`
- TQ-SFT training:
  `/home/ning/CodeClash/backup/CodeClash/colab/qwen_sft/train_flexi_weighted.py`
- Inference format test:
  `/home/ning/CodeClash/backup/CodeClash/qwen/test_flexi.py`
- Base BattleSnake dataset construction:
  `/home/ning/CodeClash/qwen/build_battlesnake_base_dataset_8agents_r15.py`
- Phase-1 metadata annotation for trajectory-quality signals:
  `/home/ning/CodeClash/qwen/statistics/annotate_phase1_metadata_top5.py`
- Phase-2 ReAct annotation with external API models:
  `/home/ning/CodeClash/qwen/annotate_phase2_claude_with_gpt5.py`
- Safe tournament runner that tolerates failed rounds:
  `/home/ning/CodeClash/qwen/run_s1000_safe.py`
- Example experiment commands:
  `/home/ning/CodeClash/experiment.sh`

## 1. Installation Guide

Recommended environment:

- Python `3.12`
- CUDA-enabled PyTorch
- `unsloth` + QLoRA training stack
- `uv` for project-level scripts

Create a clean environment and install the training dependencies:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install --upgrade --index-url https://download.pytorch.org/whl/cu128 torch==2.10.0+cu128
python -m pip install --upgrade \
  transformers==4.57.6 \
  xformers==0.0.35 \
  unsloth==2026.1.4 \
  unsloth_zoo==2026.1.4 \
  peft \
  trl==0.24.0 \
  datasets \
  accelerate \
  tensorboard \
  sentencepiece \
  openai \
  anthropic
```

For CodeClash tournament runs and data-processing scripts:

```bash
cd /home/ning/CodeClash
uv sync
```

Recommended runtime environment variables for long-context training:

```bash
export HF_HOME=/workspace/.cache/huggingface
export HF_HUB_CACHE=/workspace/.cache/huggingface
export TRANSFORMERS_CACHE=/workspace/.cache/huggingface
export XDG_CACHE_HOME=/workspace/.cache/xdg
export TMPDIR=/workspace/tmp
export WANDB_DISABLED=true
export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

## 2. Data Preprocessing

### 2.1 Build the 8-agent BattleSnake base datasets

This script extracts completed round-level trajectories for the eight teacher agents used in the report.

```bash
cd /home/ning/CodeClash/backup/CodeClash/qwen/submission
python3.12 build_battlesnake_base_dataset_8agents_r15.py \
  --root /home/ning/CodeClash/codeclash_completed_full \
  --out-dir /home/ning/CodeClash/qwen/data \
  --require-p2
```

Main outputs:

- `qwen/data/base_claudesonnet45_battlesnake_r15complete.jsonl`
- `qwen/data/base_gpt5_battlesnake_r15complete.jsonl`
- `qwen/data/base_gpt5mini_battlesnake_r15complete.jsonl`
- `qwen/data/base_grokcodefast_battlesnake_r15complete.jsonl`
- `qwen/data/base_o3_battlesnake_r15complete.jsonl`
- `qwen/data/base_claudesonnet4_battlesnake_r15complete.jsonl`
- `qwen/data/base_qwen3coderplus_battlesnake_r15complete.jsonl`
- `qwen/data/base_gemini25pro_battlesnake_r15complete.jsonl`

### 2.2 Add phase-1 metadata for TQ-SFT

This step computes rule-based action tags such as modify/check/submit and attaches a sequence-quality score under `meta.phase1`.

Note: the current `annotate_phase1_metadata_top5.py` implementation is a fixed pipeline script rather than a general CLI. It reads the predefined model-to-file mapping in the source and writes outputs under `qwen/data`.

```bash
cd /home/ning/CodeClash/backup/CodeClash/qwen/submission
python3.12 annotate_phase1_metadata_top5.py
```

Use the resulting phase-1 annotated JSONL as input to TQ-SFT.

### 2.3 Rewrite trajectories into ReAct format

This script rewrites assistant steps into `[Obs][Thought][Act]` style supervision using an API model such as Claude or GPT.

```bash
cd /home/ning/CodeClash/backup/CodeClash/qwen/submission
export ANTHROPIC_API_KEY=YOUR_KEY
python3.12 annotate_phase2_claude_with_gpt5.py \
  --jsonl-input /home/ning/CodeClash/qwen/data/base_claudesonnet45_battlesnake_r15complete.jsonl \
  --model claude-sonnet-4-5 \
  --start-row 1 \
  --limit 430
```

Example output:

- `qwen/data/base_claudesonnet45_battlesnake_r15complete_edited_claude_r1-430.jsonl`

## 3. Training

All training runs use QLoRA and assistant-only supervision.

### 3.1 Vanilla SFT

Self-play / vanilla SFT training uses `train_flexi.py`.

```bash
cd /home/ning/CodeClash/backup/CodeClash/qwen/submission
python3.12 -u train_flexi.py \
  --data /home/ning/CodeClash/qwen/data/base_qwen_battlesnake.jsonl \
  --output_dir /home/ning/CodeClash/outputs/vanilla_selfplay_sft \
  --window_turns 6 \
  --window_stride 3 \
  --max_seq_length 12000 \
  --batch_size 1 \
  --gradient_accumulation_steps 8 \
  --epochs 1 \
  --lr 1e-5 \
  --loss_reduction sample_mean
```

### 3.2 ReAct SFT

ReAct SFT uses the same training script but swaps in the phase-2 rewritten dataset.

```bash
cd /home/ning/CodeClash/backup/CodeClash/qwen/submission
python3.12 -u train_flexi.py \
  --data /home/ning/CodeClash/qwen/data/base_claudesonnet45_battlesnake_r15complete_edited_claude_r1-430.jsonl \
  --output_dir /home/ning/CodeClash/outputs/react_6turn \
  --window_turns 6 \
  --window_stride 3 \
  --max_seq_length 32000 \
  --batch_size 1 \
  --gradient_accumulation_steps 8 \
  --epochs 2 \
  --lr 1e-5 \
  --loss_reduction sample_mean
```

### 3.3 TQ-SFT

TQ-SFT uses the weighted training script and phase-1 metadata.

```bash
cd /home/ning/CodeClash/backup/CodeClash/qwen/submission
python3.12 -u train_flexi_weighted.py \
  --data /home/ning/CodeClash/qwen/data/base_top5_phase1meta_battlesnake_r15complete_merged.jsonl \
  --output_dir /home/ning/CodeClash/outputs/tq_sft \
  --window_turns 6 \
  --window_stride 6 \
  --max_seq_length 12000 \
  --batch_size 1 \
  --gradient_accumulation_steps 8 \
  --epochs 1 \
  --lr 1e-5 \
  --loss_reduction sample_mean \
  --use_phase1_weights
```

## 4. Testing

### 4.1 Quick inference-format test

This checks whether the finetuned model still produces the expected `THOUGHT` plus exactly one bash block.

```bash
cd /home/ning/CodeClash/backup/CodeClash/qwen/submission
python3.12 test_flexi.py \
  --data /home/ning/CodeClash/qwen/data/base_claudesonnet45_battlesnake_r15complete_edited_claude_r1-430.jsonl \
  --model_name /path/to/merged/model \
  --max_samples 5 \
  --max_turns_per_sample 1 \
  --temperature 0.0 \
  --print_outputs
```

### 4.2 Tournament evaluation

This runner patches empty simulator outputs so failed rounds do not terminate the entire tournament.

```bash
cd /home/ning/CodeClash/backup/CodeClash/qwen/submission
python3.12 run_s1000_safe.py \
  /home/ning/CodeClash/qwen/run_sft_6turns_vs_qwen30b_battlesnake_r15_s1000.yaml \
  -s run_sft_6turns_vs_qwen30b_battlesnake_r15_s1000
```

Other report evaluations use the same pattern with different YAML configs, for example:

- `qwen/run_sft_6turns_vs_qwenplus_battlesnake_r15_s1000.yaml`
- `qwen/run_sft_6turns_vs_claude45_battlesnake_r15_s1000.yaml`
- `qwen/run_sft_6turns_vs_weighted1k_battlesnake_r15_s1000.yaml`

## Notes

- Do not hard-code API keys or HF tokens into scripts.
- The tournament YAML files and dataset JSONL files are not duplicated into this folder.
- `experiment.sh` contains historical scratch commands; use the commands in this README as the cleaned submission reference instead.
