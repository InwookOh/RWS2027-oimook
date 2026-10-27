#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SESSION_NAME="${SESSION_NAME:-qb-vitl14-frozen-local-e5}"
RUN_TAG="$(TZ=Asia/Seoul date +%Y%m%d_%H%M%S_KST)"
EXPECTED_GPU_UUID="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["expected_gpu_uuid"])' "$PROJECT_ROOT/configs/frozen_localattr_e5.json")"
RUN_DIR="$PROJECT_ROOT/experiments/frozen_localattr_e5_$RUN_TAG"
PREFLIGHT_REPORT="$PROJECT_ROOT/reports/gpu_preflight_$RUN_TAG.json"
PREFLIGHT_LOG="$PROJECT_ROOT/reports/gpu_preflight_$RUN_TAG.log"
CONSOLE_LOG="$PROJECT_ROOT/reports/console_$RUN_TAG.log"

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "Refusing to reuse existing tmux session: $SESSION_NAME" >&2
    exit 2
fi

VISIBLE_GPU_MEMORY="$(timeout 20s nvidia-smi --id="$EXPECTED_GPU_UUID" --query-gpu=memory.used --format=csv,noheader,nounits)"
VISIBLE_GPU_APPS="$(timeout 20s nvidia-smi --id="$EXPECTED_GPU_UUID" --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits)"
if [ "$(printf '%s\n' "$VISIBLE_GPU_MEMORY" | wc -l)" -ne 1 ] || \
   [ "${VISIBLE_GPU_MEMORY%%$'\n'*}" -gt 512 ] || [ -n "$VISIBLE_GPU_APPS" ]; then
    echo "GPU is occupied or the visible GPU set is ambiguous; no tmux session started." >&2
    exit 3
fi

COMMAND="set -euo pipefail; cd $PROJECT_ROOT; export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$EXPECTED_GPU_UUID; python -u scripts/preflight.py --device cuda --batch-size 32 --report $PREFLIGHT_REPORT 2>&1 | tee $PREFLIGHT_LOG; python -u scripts/train_frozen.py --config configs/frozen_localattr_e5.json --run-dir $RUN_DIR --preflight-report $PREFLIGHT_REPORT 2>&1 | tee $CONSOLE_LOG"
tmux new-session -d -s "$SESSION_NAME" -c "$PROJECT_ROOT" "bash -lc \"$COMMAND\""
echo "Started guarded preflight/training session: $SESSION_NAME"
echo "Run directory: $RUN_DIR"
