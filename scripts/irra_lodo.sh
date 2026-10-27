#!/bin/bash
# LODO runs of the teammate's IRRA-QB FrozenMultiScaleLocalAttr model (see scripts/irra_lodo.py).
#   scripts/irra_lodo.sh setup              # once: working copies of third_party/irra (stage 2, commit 9a293b5)
#                                           # and third_party/irra_stage1 (stage 1, commit 03722b3), CLIP weights, BPE vocab
#   scripts/irra_lodo.sh run GPU DOMAIN     # stage 1 (2 epochs) -> stage 2 (2 epochs) -> held-out cache
set -e
cd "$(dirname "$0")/.."
HEAD_REPO=outputs/third_party/irra
OLD=outputs/third_party/irra-old
QB=$OLD/IRRA-QueryBalanced-ViT-L14
BPE=https://raw.githubusercontent.com/openai/CLIP/main/clip/bpe_simple_vocab_16e6.txt.gz
download() { python -c "import sys, urllib.request; urllib.request.urlretrieve(sys.argv[1], sys.argv[2])" "$1" "$2"; }
CLIP=https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt

if [ "$1" = setup ]; then
  # The teammate's code is vendored in third_party/; the
  # patches below go to working copies, so third_party/ stays as published.
  [ -d $HEAD_REPO ] || { mkdir -p $(dirname $HEAD_REPO); cp -r third_party/irra $HEAD_REPO; }
  [ -d $QB ] || { mkdir -p $OLD; cp -r third_party/irra_stage1/IRRA-QueryBalanced-ViT-L14 $QB; }
  mkdir -p $QB/weights $QB/data $HEAD_REPO/data
  # datasets/ (IRRA's image transforms) was git-ignored at 03722b3; commit 9a293b5 ships it.
  [ -d $QB/datasets ] || cp -r $HEAD_REPO/datasets $QB/datasets
  # Our LODO data roots link the image folders (symlinks); the original path check resolves
  # symlinks and rejects them. Check containment without resolving (local checkouts only).
  for f in $QB/upar/data.py $HEAD_REPO/upar/data.py; do
    sed -i 's|image_path = (root / relative).resolve()$|image_path = (root / relative).absolute()  # LODO: symlinked image folders|' $f
    grep -q "LODO: symlinked image folders" $f || { echo "could not patch $f" >&2; exit 1; }
  done
  [ -f $QB/weights/ViT-L-14.pt ] || download $CLIP $QB/weights/ViT-L-14.pt
  echo "b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836  $QB/weights/ViT-L-14.pt" | sha256sum -c
  for d in $QB/data $HEAD_REPO/data; do [ -f $d/bpe_simple_vocab_16e6.txt.gz ] || download $BPE $d/bpe_simple_vocab_16e6.txt.gz; done
  # torch >= 2 rejects a dot-prefixed file name without extension in torch.save: rename the
  # temporary weight file of the original checkpoint code (still published atomically by os.link).
  sed -i "s|output / ('.weights-tmp-' + uuid.uuid4().hex)|output / ('weights-tmp-' + uuid.uuid4().hex + '.pth')|" $QB/upar/checkpoint.py
  grep -q "'weights-tmp-' + uuid.uuid4().hex + '.pth'" $QB/upar/checkpoint.py || { echo "could not patch $QB/upar/checkpoint.py" >&2; exit 1; }
  echo "setup done"
  exit 0
fi

[ "$1" = run ] || { echo "usage: $0 setup | run GPU DOMAIN" >&2; exit 2; }
gpu=$2; domain=$3
out=outputs/k8s/irra/lodo-$domain
[ -d outputs/k8s/irra/data-$domain ] || python scripts/irra_lodo.py prepare \
    --source outputs/k8s/attrivision/data-$domain --out outputs/k8s/irra/data-$domain
data=$(realpath outputs/k8s/irra/data-$domain)
run=$QB/experiments/lodo-$domain
mkdir -p $out
uuid=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i $gpu)

if [ ! -f $run/epoch_002.pth ]; then
  # Stage 1 with the original code; stop it (gracefully, SIGINT) once epoch 2 is saved and logged.
  (cd $QB && exec python -u train_upar.py --output experiments/lodo-$domain --data-root $data \
      --pretrained weights/ViT-L-14.pt --config configs/train_fp32_micro4.json --gpu-uuid $uuid) \
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
CUDA_VISIBLE_DEVICES=$gpu python scripts/irra_lodo.py export --checkpoint $out/epoch_002.pth \
    --reference outputs/k8s/dinov2/lodo-$domain/holdout_predictions_epoch4.npz \
    --out $out/holdout_predictions_epoch2.npz
echo "done: $out/holdout_predictions_epoch2.npz"
