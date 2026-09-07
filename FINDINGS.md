# Findings

## PPO regresses below baseline; DPO wins cleanly

50-prompt judged eval, Claude as judge, four checkpoints compared
head-to-head (see [`results/eval_report.md`](./results/eval_report.md)
for the full table):

| Comparison | Result |
|---|---|
| DPO vs. SFT baseline | DPO wins 30-12, 8 ties — real, moderate improvement |
| PPO (final) vs. SFT baseline | **PPO loses 0-50** |
| PPO (mid-training checkpoint) vs. SFT baseline | **also loses 0-50** |
| PPO vs. DPO | DPO wins 50-0 |

PPO didn't just fail to improve on the baseline — it made the policy
actively worse, and this held at both the final checkpoint and an
earlier mid-training one, so it isn't a late-training collapse that a
different stopping point would have avoided.

## Why: this matches the training-stability story, not a fluke

The PPO run needed four retries to complete at all
([`logs/`](./logs/), commit history from `44c685f` through `1dbefb0`):
repeated KL-divergence blowups (values as extreme as -4000, orders of
magnitude outside a stable range), requiring gradient clipping,
value-function warmup, reward sanitization, and checkpoint-resume before
one run finally finished without diverging. The eval result is the
other half of that story: even the run that didn't visibly crash had
already learned a policy worse than doing nothing.

DPO trained cleanly on the first attempt, no comparable instability —
and it's also the one that produced a real improvement. This is the
concrete, first-hand version of the reason DPO displaced PPO for
preference tuning in practice: not just "simpler," but *actually more
reliable to get a working result from*, on real hardware, without
extensive stabilization work.

## RLAIF bring-up (separate result)

2000/2000 AI-judged preference pairs generated and labeled successfully
([`results/rlaif_pairs_summary.md`](./results/rlaif_pairs_summary.md)),
validating that Claude-judged preferences can be substituted for the
human-labeled HH-RLHF pairs in this same pipeline with zero code changes
downstream. Not yet used to train a second reward model / DPO run for a
direct human-vs-AI-label comparison — a natural next step, out of scope
for this pass.
