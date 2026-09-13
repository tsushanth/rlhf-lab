#!/usr/bin/env bash
set -euo pipefail

# Comprehensive ablation run for the RLHF lab.
#
# What this does (in order):
#   1. Check prerequisites (ANTHROPIC_API_KEY for judged eval)
#   2. Evaluate SFT vs base model using automated metrics
#   3. Train a stronger reward model (3 epochs)
#   4. Run PPO against the stronger RM
#   5. Run DPO with a beta sweep (0.05, 0.10, 0.20)
#   6. Run DPO on the existing RLAIF preference pairs
#   7. Full judged eval (base model + all checkpoints, dual judges)
#   8. Automated metrics across all checkpoints
#
# Time estimate on 1x A100 80GB:
#   SFT metrics:     5 min
#   RM (3 epochs):   45 min
#   PPO (500 steps): 90 min
#   DPO ×3 betas:    90 min total
#   DPO RLAIF:       30 min
#   Eval (50 prompts, dual judge): ~20 min
#   Metrics:         10 min
#   TOTAL:           ~4.5 hours
#
# Usage:
#   export ANTHROPIC_API_KEY=sk-ant-...
#   ./scripts/run_ablations.sh [results_dir]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

RESULTS_DIR="${1:-$REPO_ROOT/results}"
mkdir -p "$RESULTS_DIR"

# --- Prerequisites ----------------------------------------------------------
if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
  echo "[WARN] ANTHROPIC_API_KEY not set. Judged eval will fail."
  echo "       Set it now or Ctrl-C and re-run."
  sleep 3
fi

PYTHON="${PYTHON:-python3}"
echo "[info] Python: $PYTHON"
echo "[info] Results dir: $RESULTS_DIR"

# --- Step 0: SFT already trained? -------------------------------------------
if [ ! -f "$RESULTS_DIR/sft/config.json" ]; then
  echo "[error] SFT checkpoint not found at $RESULTS_DIR/sft/"
  echo "        Run first: $PYTHON scripts/01_sft.py"
  exit 1
fi

# --- Step 1: Automated metrics on SFT vs base -------------------------------
echo ""
echo "=== [1/8] Automated metrics: base vs SFT ==="
echo "    (If SFT is worse than base, the whole pipeline starts from a degraded policy.)"
$PYTHON "$SCRIPT_DIR/05b_automated_metrics.py" \
  --include-base \
  --checkpoints "sft=$RESULTS_DIR/sft" \
  --results-dir "$RESULTS_DIR" \
  --n-prompts 100 \
  --perplexity-samples 200

# Quick human-readable check
if [ -f "$RESULTS_DIR/automated_metrics.md" ]; then
  echo ""
  echo "--- Top-line metrics ---"
  grep -E "^\| base \|^\| sft \|" "$RESULTS_DIR/automated_metrics.md" || true
  echo ""
fi

# --- Step 2: Train reward model (3 epochs) -----------------------------------
echo "=== [2/8] Training reward model (3 epochs) ==="
$PYTHON "$SCRIPT_DIR/02_reward_model.py" \
  --epochs 3 \
  --results-dir "$RESULTS_DIR"

# --- Step 3: Run PPO against stronger RM ------------------------------------
echo ""
echo "=== [3/8] PPO (500 steps) against 3-epoch reward model ==="
echo "    Using 03_ppo_enhanced.py with generation snapshots every 50 steps."
$PYTHON "$SCRIPT_DIR/03_ppo_enhanced.py" \
  --steps 500 \
  --results-dir "$RESULTS_DIR" \
  --save-generations-every 50 \
  --generations-out "$RESULTS_DIR/ppo_generations.jsonl"

# --- Step 4: DPO beta sweep --------------------------------------------------
echo ""
echo "=== [4/8] DPO beta sweep: 0.05, 0.10, 0.20 ==="
for BETA in 0.05 0.10 0.20; do
  OUT_NAME="dpo_beta_${BETA}"
  echo "  -> beta=$BETA -> $RESULTS_DIR/$OUT_NAME"
  $PYTHON "$SCRIPT_DIR/04_dpo.py" \
    --beta "$BETA" \
    --epochs 1 \
    --out-name "$OUT_NAME" \
    --results-dir "$RESULTS_DIR"
done

# --- Step 5: DPO on RLAIF data -----------------------------------------------
echo ""
echo "=== [5/8] DPO on RLAIF preference pairs ==="
if [ -f "$RESULTS_DIR/rlaif_preference_pairs.jsonl" ]; then
  $PYTHON "$SCRIPT_DIR/04_dpo_rlaif.py" \
    --epochs 1 \
    --beta 0.10 \
    --results-dir "$RESULTS_DIR" \
    --compare-human-dpo
else
  echo "[skip] RLAIF pairs not found. Run 06 + 07 first."
  echo "       python scripts/06_generate_rlaif_candidates.py"
  echo "       python scripts/07_judge_rlaif_pairs.py"
fi

# --- Step 6: Judged eval with dual judges ----------------------------------
echo ""
echo "=== [6/8] Full judged eval (dual judges, base model included) ==="
echo "    This costs API credits (Claude + GPT-4o-mini)."
if [ -n "${ANTHROPIC_API_KEY:-}" ]; then
  $PYTHON "$SCRIPT_DIR/05_eval_v2.py" \
    --n-prompts 50 \
    --judge both \
    --include-base \
    --auto-metrics \
    --results-dir "$RESULTS_DIR" \
    --out-suffix "ablations"
else
  echo "[skip] No ANTHROPIC_API_KEY — skipping judged eval."
fi

# --- Step 7: Automated metrics on ALL checkpoints --------------------------
echo ""
echo "=== [7/8] Automated metrics on all checkpoints ==="
CKPTS=(
  "base=$REPO_ROOT"  # will be resolved by 05b_automated_metrics.py --include-base
)
# Build explicit checkpoint list
CKPT_LIST=()
for name in sft dpo dpo_beta_0.05 dpo_beta_0.10 dpo_beta_0.20 dpo_rlaif ppo; do
  if [ -f "$RESULTS_DIR/$name/config.json" ]; then
    CKPT_LIST+=("$name=$RESULTS_DIR/$name")
  fi
done

if [ ${#CKPT_LIST[@]} -gt 0 ]; then
  $PYTHON "$SCRIPT_DIR/05b_automated_metrics.py" \
    --include-base \
    --checkpoints "${CKPT_LIST[@]}" \
    --reward-model "$RESULTS_DIR/reward_model" \
    --results-dir "$RESULTS_DIR" \
    --n-prompts 100 \
    --perplexity-samples 200
else
  echo "[warn] No trained checkpoints found for automated metrics."
fi

# --- Step 8: Summary ---------------------------------------------------------
echo ""
echo "=== [8/8] Summary ==="
echo ""
echo "Checkpoints:"
for name in sft dpo dpo_beta_0.05 dpo_beta_0.10 dpo_beta_0.20 dpo_rlaif ppo; do
  if [ -f "$RESULTS_DIR/$name/config.json" ]; then
    echo "  ✓ $name"
  else
    echo "  ✗ $name (missing)"
  fi
done

echo ""
echo "Reports:"
for f in "$RESULTS_DIR"/eval_report*.md "$RESULTS_DIR"/automated_metrics.md; do
  [ -f "$f" ] && echo "  - $f"
done

echo ""
echo "Done. Full ablation results in $RESULTS_DIR"
