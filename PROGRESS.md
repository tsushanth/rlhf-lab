# Experiment Progress Tracker

Last updated: 2026-09-13 — starting RLAIF DPO run

## Completed ✅

| Item | Result | Notes |
|------|--------|-------|
| SFT warm-start | ✅ | `results/sft/` — 1 epoch on HH-RLHF chosen |
| Reward model (1 epoch) | ✅ | `results/reward_model/` — baseline RM |
| PPO (4 retries, 1 finish) | ✅ | `results/ppo/` — unstable, regressed to 0-50 loss vs SFT |
| DPO (beta=0.1) | ✅ | `results/dpo/` — clean run, won 30-12 vs SFT |
| Eval (50 prompts, Claude judge) | ✅ | `results/eval_report.md` — DPO >> PPO |
| RLAIF pipeline (06 → 07) | ✅ | `results/rlaif_preference_pairs.jsonl` — 2000 pairs, Claude-judged |
| Infrastructure scripts | ✅ | Committed: `04_dpo_rlaif.py`, `05b_automated_metrics.py`, `05_eval_v2.py`, `03_ppo_enhanced.py`, `run_ablations.sh` |

## In Progress ⏳

| Priority | Experiment | Script | Status |
|----------|-----------|--------|--------|
| **P0** | DPO on RLAIF data | `python scripts/04_dpo_rlaif.py --compare-human-dpo` | **STARTING NOW** |

## Pending 🔲

| Priority | Experiment | Script | Status |
|----------|-----------|--------|--------|
| **P0** | Automated metrics (base vs SFT) | `python scripts/05b_automated_metrics.py --include-base` | NOT RUN |
| **P1** | Seed variance: DPO ×3 seeds | Run `04_dpo.py` with different `--seed` (needs manual seeding in script first) | NOT RUN |
| **P1** | DPO beta sweep | `./scripts/run_ablations.sh` step 4 | NOT RUN |
| **P1** | Stronger RM → PPO retry | `./scripts/run_ablations.sh` step 2-3 | NOT RUN |
| **P1** | Dual-judge eval | `./scripts/run_ablations.sh` step 6 | NOT RUN |
| **P2** | WandB integration | Add `wandb` logging to all training scripts | NOT RUN |
| **P2** | Notebook: visualize eval results | `notebooks/` is empty | NOT RUN |
| **P2** | Unit tests for `common.py` | `pytest` for `split_prompt_response` edge cases | NOT RUN |

## Key Hypotheses to Test

1. **RLAIF ≈ human labels?** Does DPO trained on Claude-judged pairs perform similarly to DPO on human-judged pairs?
2. **PPO failure = weak RM or intrinsic?** Does PPO against a 3-epoch RM (stronger grader) still collapse?
3. **Beta sensitivity** Does DPO's win-rate hold across β ∈ {0.05, 0.10, 0.20}?
4. **Base model baseline** Is the SFT warm-start itself an improvement over raw Qwen2.5-1.5B-Instruct?

## Cost Tracker

| Run | GPU Time (A100 est.) | API Calls (judge) | Notes |
|-----|----------------------|-------------------|-------|
| Original pipeline | ~3.5 hrs | 300 (Claude) | PPO failed, DPO succeeded |
| Full ablation suite | ~4.5 hrs | 600 (Claude + GPT-4o-mini) | Includes beta sweep + RLAIF DPO |
| **RLAIF DPO** | **~30 min** | **0** | Using existing AI-labeled pairs |
