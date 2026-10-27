# UPAR 2027 Track 2: team oimook

Code for our final submission to Track 2 (attribute-based person retrieval) of the UPAR 2027
Challenge at the WACV 2027 RWS workshop (Codabench 18196, account `oimook`).

| Submission | mADM | mAP | Rank-1 | Rank-5 | Rank-10 | mINP |
|---|---:|---:|---:|---:|---:|---:|
| **Final**: six-model ensemble "C" with gallery neighbour smoothing | **0.6939** | 0.4189 | 0.4339 | 0.6356 | 0.7123 | 0.2932 |
| Previous best: ensemble "B" without smoothing | 0.6860 | 0.4097 | 0.4415 | 0.6455 | 0.7196 | 0.2889 |

## Method in short

Six attribute models predict the 40 attribute probabilities of every gallery image; a query
(a 40-bit attribute vector) ranks the gallery by a calibrated likelihood distance plus a CLIP
image-text similarity, smoothed over each image's mutual nearest neighbours in the gallery.

| # | Member | Backbone (initialization) | Input | Recipe |
|---|---|---|---|---|
| 1 | MetaCLIP text-prior head | ViT-L/14 (MetaCLIP 2.5B) | 252x126 | head initialized from CLIP text embeddings of the attributes; train + val; epoch 5, WiSE-FT alpha 0.4 with epoch 0 |
| 2 | Category-query head | ViT-L/14@336 (OpenAI CLIP) | 336x168 | CLS head plus 12 category queries over the patch tokens; LP-FT; train + val; epoch 6, WiSE-FT alpha 0.4 with epoch 1 |
| 3 | DINOv2 | ViT-L/14 (DINOv2) | 252x126 | LP-FT; train only; epoch 4, WiSE-FT alpha 0.8 with epoch 1 |
| 4 | PE-Core | ViT-L/14 (Perception Encoder) | 252x126 | LP-FT; train + val; epoch 7, WiSE-FT alpha 0.8 with epoch 1 |
| 5 | Attribute-prompt model (`e1` in the code) | ViT-L/14 (OpenAI CLIP) | 224x224 | image and text towers trained against 52 category-state prompts (2 epochs), then a frozen-encoder AttriQuery local head (3 epochs) |
| 6 | IRRA | ViT-L/14 (MetaCLIP 2.5B) | 378x126 | IRRA image-text fine-tuning of both towers (2 epochs), then a multiscale attribute branch on the frozen backbone (2 epochs) |

Members 1-4 share one recipe (`train.py`): DeepMAR-weighted BCE with label smoothing 0.1, head
dropout 0.7, AugMix, EMA 0.999, AdamW at 5e-6, effective batch 32, mixed precision, LP-FT (member 1:
text-initialized head instead) and WiSE-FT. Members 5 and 6 were developed by teammates; their code
is in `third_party/`. Inside the submission ZIP the members are stored in the order MetaCLIP,
query head, attribute-prompt model, DINOv2, PE-Core, IRRA (`model_0.pt` ... `model_5.pt`).

Retrieval score for query q and gallery image g (`upar/inference.py`):

1. label-smoothing correction of members 1-4: p' = (p - 0.025) / 0.95;
2. calmix: 0.5 * BCE(q, mean probability) + 0.5 * (-4 log mean over members of exp(-BCE_m(q) / 4));
3. minus 5 * cos(IRRA text embedding of q's attribute sentence, IRRA image embedding of g);
4. gallery neighbour smoothing: each image's 20 nearest gallery images by cosine of the DINOv2
   member's head input (its fine-tuned class token), mutual neighbours only;
   S'(g) = S(g) + 0.6 / 20 * sum over g's mutual neighbours j of (S(j) - S(g)).
   No labels or parameters are estimated on the test set; K and W were chosen on LODO.

Model selection used leave-one-domain-out validation (train on two of Market1501 / PA100k /
PETA, score the third); see `docs/EXPERIMENTS.md`.

## Repository

| Path | Content |
|---|---|
| `train.py`, `configs/` | training of members 1-4 (`configs/*_val.json`: train + validation images) |
| `upar/` | models (`vit.py`, `e1.py`, `irra.py`), data, metrics, retrieval, inference engine |
| `scripts/` | WiSE-FT soups, LODO evaluation, the attribute-prompt and IRRA pipelines and their conversion to members |
| `build_submission.py`, `submission/run.py`, `verify_submission.py` | the Codabench ZIP and an offline check |
| `third_party/` | the teammates' attribute-prompt and IRRA code, unchanged |
| `docs/EXPERIMENTS.md` | validation protocol, results, the choice of the final smoothing, and follow-ups that were not submitted |
| `TRAINING.md` | step-by-step reproduction |

## Setup

Python 3.11, PyTorch 2.4.1, torchvision 0.19.1 (the challenge runtime). We trained on RTX 2080 Ti
(11 GB) GPUs with CUDA 11.8 wheels.

```bash
conda create -n upar python=3.11 && conda activate upar
pip install -r requirements.txt
python download_datasets.py        # official train/val data into data/ (organizers' script)
```

Or with Docker (the challenge runtime: PyTorch 2.4.1, CUDA 12.1):

```bash
docker build -t upar2027-oimook .
docker run --gpus all --shm-size 16g -v $PWD/data:/workspace/data -v $PWD/outputs:/workspace/outputs \
    -it upar2027-oimook bash
```

Pretrained backbones are downloaded on first use and checked against pinned sha256 digests
(`upar/pretrained.py`, cached in `outputs/pretrained/`, or `UPAR_PRETRAINED_DIR`).

## Inference with the trained weights

Trained weights: [RWS2027-oimook.zip](https://drive.google.com/file/d/1PHmF40nZiq7E9pQQxRpPywx98L8ATQmO/view?usp=sharing)
(Google Drive). Unzipped, it contains `submission/C-smooth-k20.zip`, the final ZIP submitted to
Codabench, and `checkpoints/`, the six inference checkpoints it was built from. Unzip it, place
its `checkpoints/` folder in the repository root and rebuild the submission:

```bash
python build_submission.py --checkpoint checkpoints/metaclip-val-e5-wise0.4.pt checkpoints/query-val-e6-wise0.4.pt \
    checkpoints/e1-b3.pt checkpoints/dinov2-e4-wise0.8.pt checkpoints/pe-val-e7-wise0.8.pt \
    checkpoints/irra-metaclip-e2-text.pt --int8-weights --retrieval bce --amp-inference \
    --unsmooth 0.05 --mixture 4 0.5 --irra-text 5 --smooth 20 0.6 --smooth-mutual --smooth-features 3 \
    --output submission.zip
python verify_submission.py --zip submission.zip --device cuda --max-images 64
```

The ZIP built this way is the submitted one: same checkpoints, settings and inference code.
Its `run.py` follows the starter kit's interface; inference needs only torch, torchvision, numpy
and Pillow.

## Training from scratch

See `TRAINING.md`.

## Data

The data are the organizers' UPAR 2027 release (Market1501, PA-100K and PETA with unified
attributes). Please cite the UPAR papers and the source datasets when using them.
