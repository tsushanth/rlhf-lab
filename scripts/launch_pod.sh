#!/usr/bin/env bash
# Provisions a RunPod GPU pod for the rlhf-lab pipeline.
# Reads the API key from macOS Keychain — never hardcode it here.
set -euo pipefail

RUNPOD_KEY="$(security find-generic-password -s runpod-api-key -w)"
GPU_TYPE="${GPU_TYPE:-NVIDIA A40}"
POD_NAME="${POD_NAME:-rlhf-lab}"
IMAGE="${IMAGE:-runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04}"
SSH_PUBKEY="$(cat ~/.ssh/runpod_cuda.pub)"
MAX_RUNTIME_HOURS="${MAX_RUNTIME_HOURS:-6}"

echo "Deploying pod on ${GPU_TYPE}..."

RESPONSE=$(curl -s -X POST https://api.runpod.io/graphql \
  -H "Authorization: Bearer ${RUNPOD_KEY}" \
  -H "Content-Type: application/json" \
  -d @- <<JSON
{
  "query": "mutation { podFindAndDeployOnDemand(input: { cloudType: SECURE, gpuTypeId: \"${GPU_TYPE}\", gpuCount: 1, name: \"${POD_NAME}\", imageName: \"${IMAGE}\", containerDiskInGb: 40, volumeInGb: 60, volumeMountPath: \"/workspace\", ports: \"22/tcp,8888/tcp\", env: [{ key: \"PUBLIC_KEY\", value: \"${SSH_PUBKEY}\" }] }) { id imageName machineId } }"
}
JSON
)

echo "$RESPONSE" | python3 -m json.tool
POD_ID=$(echo "$RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin)['data']['podFindAndDeployOnDemand']['id'])")
echo ""
echo "Pod ID: ${POD_ID}"
echo "Track status: https://www.runpod.io/console/pods"
echo "Once running, get SSH details with:"
echo "  curl -s -X POST https://api.runpod.io/graphql -H \"Authorization: Bearer \$(security find-generic-password -s runpod-api-key -w)\" -H 'Content-Type: application/json' -d '{\"query\":\"query { pod(input: {podId: \\\"${POD_ID}\\\"}) { runtime { ports { ip isIpPublic privatePort publicPort type } } } }\"}'"

# Safety net: schedule a local auto-stop so a forgotten pod can't bill unattended.
# Runs on this Mac via `at`, not inside the pod — keeps the API key off shared infra.
# Extend it anytime with: atrm <job-id> (shown below), then relaunch a new one.
STOP_CMD="curl -s -X POST https://api.runpod.io/graphql -H \"Authorization: Bearer \$(security find-generic-password -s runpod-api-key -w)\" -H 'Content-Type: application/json' -d '{\"query\":\"mutation { podStop(input: {podId: \\\"${POD_ID}\\\"}) { id desiredStatus } }\"}' | tee -a ~/runpod_autostop.log; osascript -e 'display notification \"Pod ${POD_NAME} auto-stopped after ${MAX_RUNTIME_HOURS}h\" with title \"RunPod\"' 2>/dev/null || true"
echo "$STOP_CMD" | at "now + ${MAX_RUNTIME_HOURS} hours" 2>&1
echo ""
echo "Auto-stop scheduled: pod will be stopped in ${MAX_RUNTIME_HOURS}h unless you cancel it (atq / atrm)."
echo "Override with MAX_RUNTIME_HOURS=<n> next time you launch."
