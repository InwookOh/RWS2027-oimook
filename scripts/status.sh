#!/bin/bash
# Progress and held-out scores of every run below a folder (default: outputs/k8s).
root=${1:-outputs/k8s}
cd "$(dirname "$0")/.."
for log in $(find "$root" -name train.log | sort); do
  run=$(dirname "$log")
  state=$(grep -h "EXIT\|Traceback\|Error" "$log" | tail -1)
  [ -z "$state" ] && grep -q "^Finished" "$log" && state=done
  [ -z "$state" ] && state="running: $(grep -h 'batch' "$log" | tail -1 | cut -c1-45)"
  echo "== $run  [$state]"
  python - "$run" <<'PY'
import json, sys, os
path = os.path.join(sys.argv[1], 'history.jsonl')
if os.path.exists(path):
    for line in open(path):
        r = json.loads(line); v = r['validation']['metrics']; h = r.get('holdout')
        extra = f"  holdout {r['holdout_domain']} mADM {h['metrics']['mADM']:.4f} mAP {h['metrics']['mAP']:.3f}" if h else ''
        print(f"   e{r['epoch']} {r.get('phase', ''):12s} val mADM {v['mADM']:.4f}{extra}  ({r['seconds']/60:.1f} min)")
PY
done
