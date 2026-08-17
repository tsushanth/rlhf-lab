# RLAIF Preference Pairs — Bring-up Run Summary

First successful end-to-end run of the RLAIF pipeline (`06_generate_rlaif_candidates.py` → `07_judge_rlaif_pairs.py`), completed 2026-08-17.

## Results

- **2000/2000** candidate pairs generated from the SFT checkpoint, sampled over HH-RLHF prompts
- **2000/2000** pairs judged by Claude (position-debiased: judge sees A/B randomly swapped per pair, verdict swapped back)
- **0** empty responses, **0** identical chosen/rejected pairs — every pair reflects a real, distinct model comparison
- Average chosen-response length: 1085 chars; average rejected-response length: 1108 chars
- Chosen response was the *longer* one in 39.4% of pairs — no strong length bias toward either side, which is a reasonable sanity signal (a judge with a naive "longer = better" bias would skew well above 50%)

## Sample pair

**Prompt tail:** "...Do you know how to do this?"

**Chosen:** "Yes! Here's my version of a classic game. First, let's see if you're..."

**Rejected:** "...Let's suppose they're playing horseshoe bowling, right? Horseshoes aren't always thrown into an oval-shaped goal post..." (drifts off-topic from the actual question)

## Output

`results/rlaif_preference_pairs.jsonl` (gitignored — regenerable data, not committed) is in the same `{"chosen": ..., "rejected": ...}` transcript format as HH-RLHF's human labels, so `02_reward_model.py` / `04_dpo.py` can consume it via `--data-file` with zero code changes — this validates the whole point of the RLAIF pipeline: the AI-labeled data is a drop-in replacement for human labels.

## Notes

This run needed three rounds of fixes before completing cleanly:
1. `content[0].text` crashed on `ThinkingBlock` responses (extended thinking auto-enabling on this model/key)
2. A stray untracked file blocked `git pull` on the worker pod, silently causing stale-code + missing-dependency runs
3. Even after finding the text block correctly, ~1/3 of real judge calls hit `max_tokens` before ever emitting an answer, because thinking tokens consumed the whole budget — root-caused locally against 15 real HH-RLHF examples and fixed by explicitly disabling extended thinking (`thinking={"type": "disabled"}`) rather than just padding token limits

Next step: use this data to train a reward model / run DPO with `--data-file results/rlaif_preference_pairs.jsonl` and compare against the human-labeled HH-RLHF results already in `results/eval_report.md` (pending — PPO-eval objective not yet complete).
