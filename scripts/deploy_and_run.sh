#!/usr/bin/env bash
# Deploy the rlhf-lab repo to a remote GPU instance and run experiments.
#
# Usage:
#   ./scripts/deploy_and_run.sh <ssh-host>
#
# The ssh-host should be defined in ~/.ssh/config, e.g.:
#   Host vast-rlhf
#     HostName 1.2.3.4
#     Port 12345
#     User root
#
# What this does on the remote:
#   1. Installs git, python3-pip, pyenv (if needed)
#   2. Clones (or pulls) the rlhf-lab repo
#   3. Installs requirements
#   4. Runs the experiment sequence you select
#
# Before running, make sure:
#   - SSH key is added to the instance
#   - The instance has git access to github.com/tsushanth/rlhf-lab

set -euo pipefail

HOST="${1:-}"
if [ -z "$HOST" ]; then
  echo "Usage: $0 <ssh-host>"
  echo ""
  echo "Your SSH hosts (from ~/.ssh/config):"
  grep -E "^Host " ~/.ssh/config 2>/dev/null | awk '{print "  " $2}' | grep -v "^\*" || true
  echo ""
  echo "Example: $0 vast-rlhf"
  exit 1
fi

REPO_URL="${REPO_URL:-https://github.com/tsushanth/rlhf-lab.git}"
REMOTE_DIR="${REMOTE_DIR:-/workspace/rlhf-lab}"
PYTHON="${PYTHON:-python3}"

echo "=== Deploying to $HOST ==="
echo "Repo: $REPO_URL"
echo "Remote dir: $REMOTE_DIR"
echo ""

# Helper: run command on remote
rsh() {
  ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "$HOST" "$@"
}

# 1. Basic setup
rsh "
set -e
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq && apt-get install -y -qq git curl wget python3-pip python3-venv build-essential

# Install pyenv if not present (useful for managing Python versions)
if ! command -v pyenv &> /dev/null; then
  curl https://pyenv.run | bash || true
  export PATH=\"\$HOME/.pyenv/bin:\$PATH\"
  eval \"\$(pyenv init -)\" || true
fi

echo '[setup] Base dependencies installed.'
"

# 2. Clone or update repo
rsh "
set -e
if [ -d $REMOTE_DIR/.git ]; then
  echo '[deploy] Repo exists, pulling latest...'
  cd $REMOTE_DIR && git fetch origin && git reset --hard origin/main
else
  echo '[deploy] Cloning repo...'
  rm -rf $REMOTE_DIR
  git clone $REPO_URL $REMOTE_DIR
fi
"

# 3. Install Python dependencies
rsh "
set -e
cd $REMOTE_DIR

# Prefer venv
if [ ! -d venv ]; then
  $PYTHON -m venv venv
fi
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

# Also install huggingface-cli and login (if token available)
if command -v huggingface-cli &> /dev/null; then
  echo '[deploy] huggingface-cli already installed'
else
  pip install -q huggingface-hub
fi

echo '[deploy] Python deps installed.'
"

# 4. Check / optional HF login
if [ -n "${HF_TOKEN:-}" ]; then
  rsh "
cd $REMOTE_DIR
source venv/bin/activate
huggingface-cli login --token '$HF_TOKEN' --add-to-git-credential || true
"
fi

# 5. Prompt for experiment selection
echo ""
echo "=== What do you want to run on $HOST? ==="
echo ""
echo "  1) RLAIF DPO only          (~30 min) — P0 item"
echo "  2) Full ablation suite     (~4.5 hrs) — everything"
echo "  3) Automated metrics only  (~10 min) — cheap baseline check"
echo "  4) Custom command"
echo ""
read -p "Select (1-4): " choice

case "$choice" in
  1)
    REMOTE_CMD="
cd $REMOTE_DIR && source venv/bin/activate
export PYTHONUNBUFFERED=1
export ANTHROPIC_API_KEY='${ANTHROPIC_API_KEY:-}'
echo '>>> Running RLAIF DPO...'
python scripts/04_dpo_rlaif.py --compare-human-dpo --epochs 1 --beta 0.1
echo '>>> Done. Checkpoint saved to results/dpo_rlaif/'
"
    ;;
  2)
    REMOTE_CMD="
cd $REMOTE_DIR && source venv/bin/activate
export PYTHONUNBUFFERED=1
export ANTHROPIC_API_KEY='${ANTHROPIC_API_KEY:-}'
./scripts/run_ablations.sh
"
    ;;
  3)
    REMOTE_CMD="
cd $REMOTE_DIR && source venv/bin/activate
export PYTHONUNBUFFERED=1
python scripts/05b_automated_metrics.py --include-base --n-prompts 100 --perplexity-samples 200
"
    ;;
  4)
    echo ""
    read -p "Enter the command to run on the remote: " custom_cmd
    REMOTE_CMD="$custom_cmd"
    ;;
  *)
    echo "Invalid choice. Exiting."
    exit 1
    ;;
esac

# 6. Run it
echo ""
echo "=== Executing on $HOST ==="
echo "This will run in the foreground. Use Ctrl+C to interrupt (the remote process keeps running)."
echo ""
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -t "$HOST" "$REMOTE_CMD"

# 7. (Optional) Pull results back
echo ""
read -p "Pull results back to this machine? (y/n): " pull_results
if [ "$pull_results" == "y" ]; then
  LOCAL_RESULTS_DIR="./results_from_${HOST}_$(date +%Y%m%d_%H%M%S)"
  mkdir -p "$LOCAL_RESULTS_DIR"
  scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -r "${HOST}:${REMOTE_DIR}/results/*" "$LOCAL_RESULTS_DIR/" 2>/dev/null || echo "[warn] No results to pull (or scp failed)"
  echo "Results pulled to: $LOCAL_RESULTS_DIR"
fi

echo ""
echo "Done."
