#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUN_DIR="$(realpath -m "$1")"
PREFLIGHT_REPORT="$(realpath "$2")"
RUN_TAG="$(basename "$RUN_DIR")"
CONSOLE_LOG="$PROJECT_ROOT/reports/console_$RUN_TAG.log"
EXPECTED_GPU_UUID="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["expected_gpu_uuid"])' "$PROJECT_ROOT/configs/frozen_localattr_e5.json")"

cd "$PROJECT_ROOT"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$EXPECTED_GPU_UUID"
python -u scripts/train_frozen.py \
    --config configs/frozen_localattr_e5.json \
    --run-dir "$RUN_DIR" \
    --preflight-report "$PREFLIGHT_REPORT" 2>&1 | tee "$CONSOLE_LOG"
