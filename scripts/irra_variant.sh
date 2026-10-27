#!/bin/bash
# IRRA-QB FrozenMultiScaleLocalAttr (teammate's recipe, scripts/irra_lodo.sh) from another CLIP
# initialization, e.g. MetaCLIP L/14 (same layout, QuickGELU and tokenizer as OpenAI CLIP).
#   scripts/irra_variant.sh setup NAME WEIGHTS   # copy the stage-1 code, accept WEIGHTS (open_clip .bin)
#   scripts/irra_variant.sh run NAME GPU DOMAIN  # DOMAIN = Market1501 | PETA (LODO) | full (all data);
#                                                # add -val (e.g. PETA-val, full-val) to train on train + validation
#                                                # (data roots from attrivision_lodo.py prepare --with-val)
# Needs scripts/irra_lodo.sh setup first (original stage-1 checkout, BPE vocab).
set -e
cd "$(dirname "$0")/.."
HEAD_REPO=outputs/third_party/irra
SOURCE=outputs/third_party/irra-old/IRRA-QueryBalanced-ViT-L14
name=$2
QB=outputs/third_party/irra-$name/IRRA-QueryBalanced-ViT-L14

if [ "$1" = setup ]; then
  weights=$3
  [ -f "$weights" ] || { echo "missing $weights" >&2; exit 1; }
  if [ ! -d $QB ]; then
    mkdir -p $(dirname $QB)
    cp -r $SOURCE $QB
    rm -rf $QB/experiments $QB/weights
  fi
  mkdir -p $QB/weights
  ln -sfn "$(realpath $weights)" $QB/weights/foundation.bin
  # Accept an open_clip state dict (same key names as OpenAI's JIT archive) instead of only the
  # hash-pinned OpenAI file; everything else (shape checks, position resize) is unchanged.
  python - $QB/upar/pretrained.py <<'EOF'
import sys
path = sys.argv[1]
s = open(path).read()
old = """    digest = sha256(path)
    if digest != OFFICIAL_SHA256:
        raise ValueError('Official OpenAI ViT-L/14 hash mismatch')
    archive = torch.jit.load(str(path), map_location='cpu')
    raw = archive.state_dict()"""
new = """    digest = sha256(path)
    if path.suffix == '.bin':  # open_clip state dict (variant initialization, scripts/irra_variant.sh)
        raw = torch.load(str(path), map_location='cpu', weights_only=True)
    else:
        if digest != OFFICIAL_SHA256:
            raise ValueError('Official OpenAI ViT-L/14 hash mismatch')
        raw = torch.jit.load(str(path), map_location='cpu').state_dict()"""
if new not in s:
    assert s.count(old) == 1, 'unexpected pretrained.py'
    open(path, 'w').write(s.replace(old, new))
print('patched', path)
EOF
  # torch >= 2 rejects a dot-prefixed file name without extension in torch.save: rename the
  # temporary weight file of the original checkpoint code (still published atomically by os.link).
  sed -i "s|output / ('.weights-tmp-' + uuid.uuid4().hex)|output / ('weights-tmp-' + uuid.uuid4().hex + '.pth')|" $QB/upar/checkpoint.py
  grep -q "'weights-tmp-' + uuid.uuid4().hex + '.pth'" $QB/upar/checkpoint.py || { echo "could not patch $QB/upar/checkpoint.py" >&2; exit 1; }
  echo "setup done: $QB"
  exit 0
fi

[ "$1" = run ] || { echo "usage: $0 setup NAME WEIGHTS | run NAME GPU DOMAIN" >&2; exit 2; }
gpu=$3; domain=$4
base=${domain%-val}; suffix=${domain#$base}
if [ $base = full ] && [ -z "$suffix" ]; then
  data=$(realpath data); out=outputs/k8s/irra-$name/full
else
  [ -d outputs/k8s/irra/data-$domain ] || python scripts/irra_lodo.py prepare \
      --source outputs/k8s/attrivision/data-$domain --out outputs/k8s/irra/data-$domain
  data=$(realpath outputs/k8s/irra/data-$domain)
  if [ $base = full ]; then out=outputs/k8s/irra-$name/full$suffix; else out=outputs/k8s/irra-$name/lodo-$domain; fi
fi
run=$QB/experiments/$domain
mkdir -p $out
uuid=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i $gpu)

if [ ! -f $run/epoch_002.pth ]; then
  (cd $QB && exec python -u train_upar.py --output experiments/$domain --data-root $data \
      --pretrained weights/foundation.bin --config configs/train_fp32_micro4.json --gpu-uuid $uuid) \
      > $out/stage1.log 2>&1 &
  pid=$!
  while kill -0 $pid 2>/dev/null; do
    if [ -f $run/epoch_002.pth ] && [ "$(wc -l < $run/metrics.jsonl 2>/dev/null || echo 0)" -ge 2 ]; then
      kill -INT $pid; break
    fi
    sleep 60
  done
  wait $pid || true
fi
[ -f $run/epoch_002.pth ] || { echo "stage 1 did not reach epoch 2; see $out/stage1.log" >&2; exit 1; }

[ -f $out/epoch_002.pth ] || CUDA_VISIBLE_DEVICES=$gpu python -u scripts/irra_lodo.py stage2 --repo $HEAD_REPO \
    --stage1 $run/epoch_002.pth --data $data --out $out --epochs 2 > $out/stage2.log 2>&1
if [ $base != full ]; then
  CUDA_VISIBLE_DEVICES=$gpu python scripts/irra_lodo.py export --checkpoint $out/epoch_002.pth \
      --reference outputs/k8s/dinov2/lodo-$base/holdout_predictions_epoch4.npz \
      --out $out/holdout_predictions_epoch2.npz --globals $out/globals_epoch2.npz
fi
echo "done: $out"
