# Validation protocol and results

## Leave-one-domain-out (LODO) validation

The hidden test set comes from a domain that is not among the three training datasets, while
the official validation split shares their domains. Selecting epochs on it cost 0.02-0.03 mADM
on the server, so every choice was made with LODO instead: train on two of Market1501 /
PA100k / PETA (train + seen-domain validation images) and score the held-out domain's
validation images with the full retrieval score.

- **Splits.** We use the mean of the Market1501 and PETA held-out splits ("M+P"). Their
  training sets (88-89k images) resemble the all-domain one; holding out PA100k leaves only
  19k training images and predicted the server worse. PA100k is used as a far-domain check.
- **Rule.** A change is adopted when it raises M+P with both splits up. On the server, LODO
  gains transferred at about 40-100%; changes below about +0.003 often did not show.
- **Check.** Re-scoring 13 earlier submissions on their LODO caches ordered 95% of the pairs as
  the server did (Kendall tau-a 0.90 for M+P).

## How the submission was built (server mADM)

| Step | Server mADM |
|---|---:|
| OpenAI CLIP ViT-L/14, LP-FT, epoch 4 + WiSE-FT (single model, flip TTA) | 0.6480 |
| ensembles of CLIP members (text-prior head, category-query head, MetaCLIP) | 0.6549-0.6712 |
| query head + MetaCLIP text-prior + DINOv2 + attribute-prompt model (no TTA) | 0.6781 |
| + label-smoothing correction and likelihood mixture (calmix) | 0.6799 |
| + PE-Core | 0.6817 |
| + IRRA member (attributes) | 0.6821 |
| IRRA from MetaCLIP L/14, plus its image-text score (lambda 5) | 0.6845 |
| MetaCLIP, DINOv2 and PE retrained on train + validation images ("B") | 0.6860 |
| ensemble C (epochs and WiSE-FT alphas re-chosen on LODO) + gallery neighbour smoothing (**final**) | **0.6939** |

Final entry: mADM 0.6939, mAP 0.4189, Rank-1 0.4339, Rank-5 0.6356, Rank-10 0.7123, mINP 0.2932
(B: 0.6860 / 0.4097 / 0.4415 / 0.6455 / 0.7196 / 0.2889). mAP and mINP rose while Rank-1/5/10
fell (-0.008 / -0.010 / -0.007): a trade-off, as mutual smoothing also lowered Rank-1 on LODO.
C and smoothing were submitted together, so the server cannot separate them.

Ingredients and their LODO effect:

- **LP-FT** (linear probe, then fine-tuning): +0.023 for CLIP backbones; the MetaCLIP member
  instead fits its head to CLIP text embeddings of the attributes (`upar/text_prior.py`).
- **WiSE-FT** (weights (1 - alpha) * start + alpha * fine-tuned): large and consistent; the
  ensemble prefers lower alphas than single models.
- **BCE retrieval** (negative log-likelihood of the complete query) instead of L1 on the
  probabilities, then **calmix** (label-smoothing correction eps 0.05 and a soft-min mixture of
  the members' likelihoods, T 4, weight 0.5): +0.004.
- **IRRA image-text score:** +0.0057 on LODO, +0.0024 on the server.
- **Diversity of pretraining matters:** replacing an OpenAI-initialized member (query head, attribute-prompt model)
  by a member from another pretraining lowered the score even when the replacement was
  stronger alone.

Not adopted (no gain on LODO or on the server): listwise and smoothed-mADM ranking losses,
an ensemble-aware head, query-set normalization, SWAD, domain-balanced sampling and domain
biases, low-resolution and colour augmentation, higher input resolution, DINOv3, SigLIP 2,
DataComp / LAION / DFN CLIP, and our recipes (WiSE-FT, epoch choice) applied to the attribute-prompt model and IRRA.

## Final submission: ensemble C with gallery neighbour smoothing

Codabench was unreachable from October 21 until after the deadline; one more submission was
allowed on October 27, and the members of C were retrained for it.

**Ensemble C.** The same six members with MetaCLIP at epoch 5 / WiSE-FT 0.4, the query head
retrained on train + validation (epoch 6, 0.4), PE at epoch 7 (0.8) and the original DINOv2
(train only, epoch 4, 0.8): LODO M+P 0.7455 against about 0.743 for B. Configs
`metaclip_b7_val.json`, `b8_query_val.json`, `dinov2.json`, `pe_core_val.json`; scoring unchanged.

**Gallery neighbour smoothing** (`build_submission.py --smooth K W [--smooth-mutual]
[--smooth-features INDEX]`). The final distances of every gallery image are mixed with those
of its K nearest gallery images: a frame whose attributes were confidently mispredicted is
pulled back by other images of the same person. Most PETA queries have a single image and
Market1501 queries many, so we report all queries and queries with at least 5 or 10 images.
On the full C ensemble (gain over C, K = 10):

| Neighbours by | Weighting | all queries | >= 5 images | >= 10 images |
|---|---|---:|---:|---:|
| IRRA image embedding | (1 - W) D + W mean D_j, W 0.4 | -0.0078 | +0.0123 | +0.0181 |
| IRRA image embedding | mutual neighbours only, W 0.4 | +0.0022 | +0.0076 | +0.0097 |
| IRRA image embedding | mutual neighbours only, W 0.6 | -0.0001 | +0.0084 | +0.0124 |
| IRRA image embedding | mutual, W 0.6, K = 20 | +0.0016 | +0.0089 | +0.0135 |
| DINOv2 member features | mutual neighbours only, W 0.4 | +0.0029 | | +0.0095 |
| DINOv2 member features | mutual neighbours only, W 0.6 | +0.0013 | | +0.0123 |

The gain comes from mAP and mINP (mutual smoothing lowers Rank-1 a little): queries whose
positives were buried by one confidently wrong attribute. Mutual neighbours keep images of
people seen once untouched, so this variant gains both over all queries and on queries with
many images. Features that are too attribute-specific (the query head) or not fine-tuned at all
(pretrained DINOv2) find worse neighbours; IRRA's text-aligned embedding groups images that
share IRRA's own errors, DINOv2's appearance features less so.

**Choosing K and W (10-27).** The six-member LODO caches had been deleted, so DINOv2 and MetaCLIP
of C were retrained on the LODO folds (`outputs/c-lodo`) and the grid was scored with their
calmix distances and DINOv2 neighbours (`swap_lodo.py --neighbour-cache`, no text term). Mutual
smoothing, gain in M+P (Market1501 / PETA), all queries:

| | K 10, W 0.6 | K 20, W 0.6 | K 30, W 0.6 | K 20, W 0.4 | K 20, W 0.8 |
|---|---|---|---|---|---|
| DINOv2 alone | +0.0051 (+0.0116 / -0.0013) | +0.0055 (+0.0123 / -0.0012) | +0.0046 | +0.0052 | +0.0048 |
| DINOv2 + MetaCLIP | +0.0054 (+0.0126 / -0.0018) | +0.0052 (+0.0130 / -0.0026) | +0.0036 | +0.0054 | +0.0031 |

On queries with at least 10 images, K 10 / 20 / 30 at W 0.6 gave +0.0119 / +0.0133 / +0.0097
(DINOv2 alone) and +0.0137 / +0.0137 / +0.0092 (two members). Plain K nearest neighbours lose on
all queries at every K and W (PETA down to -0.097). Larger W keeps gaining on queries with many
images but costs PETA more, and single models, which err more than the ensemble, favour larger W:
W stays at 0.6. K = 20 is at least as good as K = 10 on Market1501 in every configuration (also
with IRRA neighbours on the full ensemble), at a cost of at most 0.001 on PETA. On the held-out
Market1501 images (about 22 per identity), DINOv2 mutual neighbours of an image number 3.3 / 6.6
/ 13.2 / 19.8 for K 5 / 10 / 20 / 30, of which 94 / 87 / 76 / 66% show the same person: K = 20
doubles the same-person neighbours of K = 10 (10.1 against 5.7), while K = 30 adds more other
people than same-person images. Submitted: mutual, K = 20, W = 0.6, DINOv2 features.

## Follow-ups not submitted

**MetaCLIP ViT-H/14** (16 frozen blocks to fit 11 GB) reached 0.700 held-out mADM alone with
WiSE-FT and +0.0018 on LODO in place of IRRA's attribute vote (IRRA kept for the text score),
at the cost of a seventh model in the ZIP.
