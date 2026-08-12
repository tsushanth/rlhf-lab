#!/usr/bin/env bash
set -e
cd /workspace/rlhf-lab/scripts
: "${ANTHROPIC_API_KEY:?Set ANTHROPIC_API_KEY before running (used by 05_eval.py for judged win-rates)}"
echo "[$(date)] Starting SFT"
python3 -u 01_sft.py --epochs 1
echo "[$(date)] Starting reward model (3 epochs)"
python3 -u 02_reward_model.py --epochs 3
echo "[$(date)] Starting DPO"
python3 -u 04_dpo.py --epochs 1
echo "[$(date)] Starting PPO (500 steps, all fixes applied)"
python3 -u 03_ppo.py --steps 500
echo "[$(date)] Starting eval"
python3 -u 05_eval.py --n-prompts 50
echo "[$(date)] PIPELINE COMPLETE"
