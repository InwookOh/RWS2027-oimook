#!/bin/bash
# Launch one AMP training run in the background on one GPU.
# usage: scripts/run.sh GPU OUTPUT_DIR CONFIG [extra train.py args, e.g. --holdout-domain PETA]
set -e
gpu=$1; out=$2; config=$3; shift 3
cd "$(dirname "$0")/.."
# Any earlier log means a run may still be writing here, possibly from another pod.
# Stop it first (pkill -f "$out"), then delete the folder before relaunching.
if [ -e "$out/train.log" ] || [ -e "$out/history.jsonl" ] || [ -e "$out/last.training.pt" ]; then
  echo "$out already has a run; stop its process and remove the folder first" >&2; exit 1
fi
mkdir -p "$out"
CUDA_VISIBLE_DEVICES=$gpu nohup python -u train.py --config "$config" --amp --workers 8 --epochs 6 "$@" \
  --output-dir "$out" > "$out/train.log" 2>&1 &
echo "GPU $gpu -> $out (pid $!)"
