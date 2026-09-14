#!/usr/bin/env bash
# Launches a Vast.ai instance for the rlhf-lab pipeline.
#
# Prerequisites:
#   pip install vastai
#   vastai set api-key YOUR_KEY
#
# Usage:
#   ./scripts/launch_vast.sh          # interactive — choose from menu
#   GPU_TYPE="RTX 4090" ./scripts/launch_vast.sh  # filter by GPU

set -euo pipefail

GPU_TYPE="${GPU_TYPE:-}"          # e.g. "RTX 4090", "A100", "A6000"
IMAGE="${IMAGE:-pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime}"
DISK_GB="${DISK_GB:-60}"
MAX_RUNTIME_HOURS="${MAX_RUNTIME_HOURS:-6}"
SSH_KEY="${SSH_KEY:-$(cat ~/.ssh/id_ed25519.pub 2>/dev/null || cat ~/.ssh/id_rsa.pub 2>/dev/null || echo "")}"

if ! command -v vastai &> /dev/null; then
  echo "[error] vastai CLI not found. Install with: pip install vastai"
  exit 1
fi

echo "Searching for available instances..."

# Build query filters
filters="cuda_vers>=12 rentable=true"
if [ -n "$GPU_TYPE" ]; then
  filters="${filters} gpu_name=='${GPU_TYPE}'"
fi

# Query offers, sorted by cheapest hourly rate
offers=$(vastai search offers "$filters" --raw 2>/dev/null)
if [ -z "$offers" ] || [ "$offers" == "[]" ]; then
  echo "[error] No matching instances found. Try a different GPU_TYPE."
  echo "Available GPUs:"
  vastai search offers "rentable=true" --raw 2>/dev/null | python3 -c "
import sys, json
gpus = set()
for o in json.load(sys.stdin):
    gpus.add(o.get('gpu_name', 'unknown'))
for g in sorted(gpus)[:20]:
    print(f'  {g}')
" || true
  exit 1
fi

# Show top 5 cheapest
echo ""
echo "Top 5 cheapest offers:"
vastai search offers "$filters" --raw 2>/dev/null | python3 -c "
import sys, json
offers = json.load(sys.stdin)
offers = sorted(offers, key=lambda x: x.get('dph_total', 999))[:5]
for i, o in enumerate(offers):
    gpu = o.get('gpu_name', '?')
    num = o.get('num_gpus', 1)
    dph = o.get('dph_total', '?')
    ram = o.get('cpu_ram', '?')
    disk = o.get('disk_space', '?')
    loc = o.get('geolocation', '?')
    id = o.get('id', '?')
    print(f'{i+1}. [{id}] {gpu} x{num} — \${dph:.3f}/h — {ram}GB RAM — {loc}')
"

# Prompt for selection unless ID is set via env
if [ -z "${INSTANCE_ID:-}" ]; then
  echo ""
  read -p "Enter number (1-5) or paste offer ID directly: " choice
  if [[ "$choice" =~ ^[0-9]+$ ]] && [ "$choice" -ge 1 ] && [ "$choice" -le 5 ]; then
    INSTANCE_ID=$(echo "$offers" | python3 -c "
import sys, json
offers = json.load(sys.stdin)
offers = sorted(offers, key=lambda x: x.get('dph_total', 999))[:5]
print(offers[$choice-1]['id'])
")
  else
    INSTANCE_ID="$choice"
  fi
fi

echo ""
echo "Creating instance from offer ${INSTANCE_ID}..."

# Create instance
vastai create instance "$INSTANCE_ID" \
  --image "$IMAGE" \
  --disk "$DISK_GB" \
  --ssh \
  --direct \
  ${SSH_KEY:+--ssh-key "$(echo "$SSH_KEY" | head -1)"} \
  --env "-e DEBIAN_FRONTEND=noninteractive" \
  --onstart-cmd "apt-get update && apt-get install -y git curl python3-pip && pip install --upgrade pip"

echo ""
echo "Instance created. Fetching connection details..."
sleep 5

# Get instance info
instance_info=$(vastai show instances --raw 2>/dev/null | python3 -c "
import sys, json
instances = json.load(sys.stdin)
# Find the most recently created running instance
for inst in sorted(instances, key=lambda x: x.get('id', 0), reverse=True):
    if inst.get('actual_status') == 'running':
        print(json.dumps(inst))
        break
" || echo "")

if [ -z "$instance_info" ]; then
  echo "[warn] Could not auto-detect new instance. Check manually:"
  echo "  vastai show instances"
  exit 0
fi

INSTANCE_IP=$(echo "$instance_info" | python3 -c "import sys,json; print(json.load(sys.stdin).get('public_ipaddr', '???'))")
INSTANCE_PORT=$(echo "$instance_info" | python3 -c "import sys,json; print(json.load(sys.stdin).get('ssh_port', '???'))")
INSTANCE_ID_FINAL=$(echo "$instance_info" | python3 -c "import sys,json; print(json.load(sys.stdin).get('id', '???'))")

echo ""
echo "Instance ready!"
echo "  ID:    $INSTANCE_ID_FINAL"
echo "  SSH:   ssh -p $INSTANCE_PORT root@$INSTANCE_IP"
echo "  HTTP:  http://$INSTANCE_IP:$INSTANCE_PORT"
echo ""
echo "Add to ~/.ssh/config:"
echo "Host vast-rlhf"
echo "  HostName $INSTANCE_IP"
echo "  Port $INSTANCE_PORT"
echo "  User root"
echo "  StrictHostKeyChecking no"
echo "  UserKnownHostsFile /dev/null"
echo ""

# Auto-destroy after MAX_RUNTIME_HOURS
STOP_CMD="vastai destroy instance $INSTANCE_ID_FINAL; osascript -e 'display notification \"Vast instance $INSTANCE_ID_FINAL destroyed after ${MAX_RUNTIME_HOURS}h\" with title \"Vast.ai\"' 2>/dev/null || true"
echo "$STOP_CMD" | at "now + ${MAX_RUNTIME_HOURS} hours" 2>/dev/null || echo "[warn] Could not schedule auto-destroy (at not installed). Remember to destroy manually: vastai destroy instance $INSTANCE_ID_FINAL"
echo ""

echo "Next step: deploy and run experiments"
echo "  ./scripts/deploy_and_run.sh vast-rlhf"
