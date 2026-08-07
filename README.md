# RLHF Lab: PPO vs DPO on HH-RLHF

A from-scratch comparison of the two dominant preference-tuning methods —
classic RLHF (reward model + PPO) and DPO (direct preference optimization) —
fine-tuning a small open model on Anthropic's HH-RLHF human preference dataset.

## Why this project

Most "RLHF" job screens want to see that you understand the full pipeline:
preference data → reward model → policy optimization, and that you understand
*why* the field mostly moved from PPO to DPO. This repo builds both paths on
the same base model and same data, so the comparison is apples-to-apples.

## Pipeline

```
Anthropic/hh-rlhf (preference pairs)
        │
        ├── SFT warm-start ─── scripts/01_sft.py
        │         │
        │         ├── Reward model ─── scripts/02_reward_model.py
        │         │         │
        │         │         └── PPO ─── scripts/03_ppo.py ──────┐
        │         │                                             │
        │         └── DPO ─── scripts/04_dpo.py ─────────────┐  │
        │                                                    │  │
        └──────────────────────────────── scripts/05_eval.py ┴──┘
                                              (compare all 3)
```

## Model

Default: `Qwen/Qwen2.5-1.5B-Instruct` — small enough to SFT/DPO/PPO in a
few hours on a single rented GPU (A10/A100), strong enough that results are
legible instead of noise. Swap via `--model` on any script.

## Setup (cloud GPU)

Rent a GPU box — RunPod, Lambda Labs, or a Colab Pro A100 all work. Cheapest
reasonable option: RunPod `1x A100 80GB` on-demand (~$1.5-2/hr), this whole
pipeline finishes in under 4 hours of compute.

```bash
pip install -r requirements.txt
huggingface-cli login   # needed to pull Qwen2.5 + push results (optional)
```

## Run order

```bash
python scripts/01_sft.py              # warm-start on chosen responses only
python scripts/02_reward_model.py     # train reward model on pairs
python scripts/03_ppo.py              # PPO against the reward model
python scripts/04_dpo.py              # DPO directly on pairs (from SFT checkpoint)
python scripts/05_eval.py             # win-rate comparison: SFT vs PPO vs DPO
```

Each script checkpoints to `results/<name>/` and is independently resumable.

## What "done" looks like

`results/eval_report.md` — win-rate table (judged by a stronger model, GPT-4o
or Claude, since human eval doesn't scale for a side project) comparing:
- SFT-only baseline
- SFT → PPO (reward-model-guided)
- SFT → DPO (direct)

Plus a short writeup of training stability differences (PPO is famously
touchier — reward hacking, KL blowups — which is half the point of building
both).

## Cost estimate

~$15-25 total in GPU rental for the full pipeline on Qwen2.5-1.5B. Scaling to
7B is possible but roughly 5-8x the compute; not necessary to make the point.
