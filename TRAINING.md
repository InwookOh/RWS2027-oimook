# Reproducing the submission

All commands run from the repository root after the setup in `README.md`. Paths under
`outputs/` are only conventions; checkpoints are written where `--output-dir`/`--out` point.
Each of the six members trains on one 11 GB GPU (RTX 2080 Ti) with mixed precision.

## 1. Members 1-4: MetaCLIP, query head, DINOv2, PE-Core (`train.py`)

`scripts/run.sh GPU OUTPUT CONFIG [train.py arguments]` starts one run in the background
(log in `OUTPUT/train.log`). `--epochs` is the epoch whose checkpoint is used; every epoch is
saved as `epochN.pt` (epoch 0 is the text-prior head of member 1, epoch 1 the linear-probe
epoch of the others).

```bash
scripts/run.sh 0 outputs/c/metaclip-val configs/metaclip_b7_val.json --epochs 5
scripts/run.sh 1 outputs/c/query-val configs/b8_query_val.json --epochs 6
scripts/run.sh 2 outputs/c/dinov2 configs/dinov2.json --epochs 4
scripts/run.sh 3 outputs/c/pe-val configs/pe_core_val.json --epochs 7
```

WiSE-FT: interpolate the chosen epoch with the start of fine-tuning.

```bash
mkdir -p outputs/soups/full
python scripts/make_soup.py outputs/c/metaclip-val/epoch5.pt --base outputs/c/metaclip-val/epoch0.pt --alpha 0.4 --output outputs/soups/full/metaclip-val-e5-wise0.4.pt
python scripts/make_soup.py outputs/c/query-val/epoch6.pt --base outputs/c/query-val/epoch1.pt --alpha 0.4 --output outputs/soups/full/query-val-e6-wise0.4.pt
python scripts/make_soup.py outputs/c/dinov2/epoch4.pt --base outputs/c/dinov2/epoch1.pt --alpha 0.8 --output outputs/soups/full/dinov2-e4-wise0.8.pt
python scripts/make_soup.py outputs/c/pe-val/epoch7.pt --base outputs/c/pe-val/epoch1.pt --alpha 0.8 --output outputs/soups/full/pe-val-e7-wise0.8.pt
```

## 2. Member 5: attribute-prompt model (`e1` in the code; `third_party/attrivision`)

Stage A trains both CLIP towers and is stopped after epoch 2 of its 100-epoch schedule; stage B
trains the local head on the frozen stage-A encoders and is stopped after epoch 3 of its 30-epoch
schedule (Gradient Cache keeps an effective batch of 128 with micro-batches of 8).

```bash
python scripts/attrivision_lodo.py e1a --data data --out outputs/k8s/e1/full/a --stop-after 2
python scripts/attrivision_lodo.py e1b --data data --init outputs/k8s/e1/full/a/checkpoint_epoch_002.pth \
    --out outputs/k8s/e1/full/b --stop-after 3
python scripts/e1_member.py convert --checkpoint outputs/k8s/e1/full/b/checkpoint_epoch_003.pth \
    --output outputs/soups/full/e1-b3.pt
```

## 3. Member 6: IRRA (`third_party/irra`, `third_party/irra_stage1`)

`irra_lodo.sh setup` makes working copies of the IRRA code under `outputs/third_party/` and
downloads the OpenAI CLIP weights and BPE vocabulary; `irra_variant.sh setup` switches the
initialization to MetaCLIP L/14 (the same file member 1 uses; it is downloaded by step 1, or run
`python -c "from upar.pretrained import fetch; print(fetch('clip_metaclip_l14'))"`).
`irra_variant.sh run` trains stage 1 (2 epochs) and stage 2 (2 epochs).

```bash
scripts/irra_lodo.sh setup
scripts/irra_variant.sh setup metaclip outputs/pretrained/clip_metaclip_l14/open_clip_pytorch_model.bin
scripts/irra_variant.sh run metaclip 0 full
python scripts/irra_member.py convert --text --checkpoint outputs/k8s/irra-metaclip/full/epoch_002.pth \
    --output outputs/soups/full/irra-metaclip-e2-text.pt
```

`--text` keeps the IRRA text tower, which the image-text term of the retrieval score needs.

## 4. Submission ZIP

```bash
F=outputs/soups/full
python build_submission.py --checkpoint $F/metaclip-val-e5-wise0.4.pt $F/query-val-e6-wise0.4.pt $F/e1-b3.pt \
    $F/dinov2-e4-wise0.8.pt $F/pe-val-e7-wise0.8.pt $F/irra-metaclip-e2-text.pt \
    --int8-weights --retrieval bce --amp-inference --unsmooth 0.05 --mixture 4 0.5 --irra-text 5 \
    --smooth 20 0.6 --smooth-mutual --smooth-features 3 --output outputs/submission.zip
python verify_submission.py --zip outputs/submission.zip --device cuda --max-images 64
```

`--int8-weights` stores large weight matrices as int8 with per-row scales (the attribute-prompt model stays float16), so
the six models fit in a 2.2 GiB ZIP; `--amp-inference` runs float16 autocast. Both keep the
scores of the float32 models within noise and the ensemble within the 300 s limit.
`--smooth 20 0.6 --smooth-mutual --smooth-features 3` adds the gallery neighbour smoothing: the
neighbours come from the head input of the fourth checkpoint in the list (DINOv2; index 3 counting
from 0) and are found on the GPU at test time. The checkpoint order is the one of the submitted ZIP. Without these three options the same command builds ensemble C without smoothing.

## 5. Leave-one-domain-out validation

All choices (members, epochs, WiSE-FT alphas, scoring parameters) were made on held-out
domains, never on the official validation split, which shares the training domains.

```bash
# a run that never sees PETA; it writes holdout_predictions_epoch<N>.npz for every epoch
scripts/run.sh 0 outputs/k8s/dinov2-val/lodo-PETA configs/dinov2_val.json --epochs 6 --holdout-domain PETA
# WiSE-FT alphas of one run on its held-out domain(s)
python scripts/soup_lodo.py "outputs/k8s/dinov2-val/lodo-{domain}:4" --base "outputs/k8s/dinov2-val/lodo-{domain}:1" \
    --alphas 0.6 0.7 0.8 0.9 --domains Market1501 PETA --output-dir outputs/soups/wise-dinov2-val-e4
# the full scoring of an ensemble, one slot per member; several versions per slot are compared
python scripts/swap_lodo.py --slot "metaclip=outputs/soups/wise-metaclip-b7-val-e4/{domain}/alpha0.6.npz" \
    --slot ... --text-cache "outputs/k8s/irra-metaclip/lodo-{domain}/globals_epoch2.npz"
```

The smoothing settings were chosen on LODO from the held-out caches and DINOv2 features of
LODO runs (`export_features.py` hooks the same head input that the submission uses):

```bash
python scripts/soup_lodo.py "outputs/c-lodo/dinov2/lodo-{domain}:4" --base "outputs/c-lodo/dinov2/lodo-{domain}:1" \
    --alphas 0.8 --domains Market1501 PETA --output-dir outputs/c-lodo/soups/dinov2-e4
python scripts/export_features.py --soup-cache "outputs/c-lodo/soups/dinov2-e4/{domain}/alpha0.8.npz" \
    --domains Market1501 PETA --output "outputs/c-lodo/features/dinov2/{domain}.npz"
python scripts/swap_lodo.py --slot "dinov2=outputs/c-lodo/soups/dinov2-e4/{domain}/alpha0.8.npz" \
    --neighbour-cache "outputs/c-lodo/features/dinov2/{domain}.npz" --smooth 10,0.6,mutual --smooth 20,0.6,mutual
```

The attribute-prompt and IRRA LODO runs use data roots without the held-out domain
(`scripts/attrivision_lodo.py prepare --domain PETA --out outputs/k8s/attrivision/data-PETA`,
then `e1a`/`e1b`/`export-e1` on that root, and `scripts/irra_variant.sh run metaclip GPU PETA`).
`docs/EXPERIMENTS.md` has the protocol and the numbers.
