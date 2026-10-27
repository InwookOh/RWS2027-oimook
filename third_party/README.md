# Teammates' code used by the submission

Two of the six ensemble members, the attribute-prompt model (`e1` in the code) and the IRRA member, were developed by teammates. Their
code is included here unchanged, as it was when the submitted members were trained, so that the
submission can be retrained from this repository alone.

| Folder | Used for |
|---|---|
| `attrivision/` | attribute-prompt model: CLIP ViT-L/14 image and text towers trained against 52 category-state prompts (stage A), then a frozen-encoder AttriQuery local head (stage B) |
| `irra_stage1/` | IRRA stage 1: query-balanced IRRA fine-tuning of both CLIP towers (SDM + masked attribute prediction) |
| `irra/` | IRRA stage 2 modules (frozen backbone, multiscale fusion and attribute branch) and image transforms |

`irra/` and `irra_stage1/` keep their `LICENSE`.

Our side of the integration lives in `scripts/`:

- `attrivision_lodo.py` runs stages A and B on a data root and exports their predictions;
  `e1_member.py` converts a stage-B checkpoint into an inference member (`upar/e1.py`).
- `irra_lodo.sh setup` makes working copies of `irra/` and `irra_stage1/` under
  `outputs/third_party/` and applies three small patches there (symlinked image folders, a
  temporary file name that recent PyTorch accepts, `datasets/` copied into the stage-1 code);
  `irra_variant.sh setup metaclip WEIGHTS` adds the MetaCLIP L/14 initialization (open_clip
  state dict) used by the submitted member; `irra_lodo.py stage2` reproduces stage 2;
  `irra_member.py` converts the result into an inference member (`upar/irra.py`).

The code in these folders is not edited (only the setup instructions in `irra/README.md` point to
this folder); patches are applied to the working copies only.
