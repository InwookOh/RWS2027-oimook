"""LODO runs of a teammate's IRRA-QB FrozenMultiScaleLocalAttr model (third_party/irra, third_party/irra_stage1).

Stage 1 (QueryBalanced IRRA fine-tuning) runs the original code of commit 03722b3 unchanged
(scripts/irra_lodo.sh). Stage 2 here reproduces the original scripts/train_frozen.py loop with
its own modules: the stage-1 epoch-2 weights are loaded, only the multiscale fusion and the
attribute branch train with 0.5 * BCE (Adam lr 1e-4, betas 0.9/0.999, eps 1e-3, weight decay
4e-5, batch 32, query-balanced sampling, cosine T_max 5 down to 1e-5 stepped per epoch, seed
20271013); validation is skipped. `export` predicts the held-out gallery with our port
(upar/irra.py) and writes a holdout_predictions_epoch<N>.npz cache for ensemble_lodo.py.

    python scripts/irra_lodo.py stage2 --repo outputs/third_party/irra \
        --stage1 <stage-1 run>/epoch_002.pth --data outputs/k8s/attrivision/data-PETA \
        --out outputs/k8s/irra/lodo-PETA --epochs 2
    python scripts/irra_lodo.py export --checkpoint outputs/k8s/irra/lodo-PETA/epoch_002.pth \
        --reference outputs/k8s/dinov2/lodo-PETA/holdout_predictions_epoch4.npz \
        --out outputs/k8s/irra/lodo-PETA/holdout_predictions_epoch2.npz
"""
import argparse
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def stage2(args):
    repo = Path(args.repo).resolve()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.glob('epoch_*.pth')):
        raise SystemExit(f'{out} already has stage-2 checkpoints')
    sys.path.insert(0, str(repo))  # the original modules (its own `upar`, `model`, `scripts`)
    from torch.utils.data import DataLoader
    from scripts.preflight import strict_load_new_model
    from upar.checkpoint import load_state_dict
    from upar.data import TrainingDataset, UPARSplit
    from upar.sampling import QueryBalancedSampler
    from upar.tokenizer import PromptTokenizer

    seed = 20271013
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = torch.device('cuda')
    split = UPARSplit.read(Path(args.data).resolve(), 'train')
    sampler = QueryBalancedSampler(split, seed)
    loader = DataLoader(TrainingDataset(split, PromptTokenizer()), batch_size=32, sampler=sampler,
                        num_workers=args.workers, generator=torch.Generator().manual_seed(seed))
    state = load_state_dict(args.stage1)
    if args.strip_heads:
        # Start from a finished stage-2 checkpoint: its frozen backbone, text tower and stage-1
        # modules equal the stage-1 weights; dropping the trained branch lets the original loader
        # re-initialize it with the same seed, as from the stage-1 checkpoint itself.
        state = {k: v for k, v in state.items() if not k.startswith(('multiscale.', 'local_branch.'))}
    model, _ = strict_load_new_model(state, seed)
    torch.cuda.manual_seed_all(seed)
    model.to(device)
    model.configure_frozen_training()
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=1e-4, betas=(0.9, 0.999), eps=1e-3, weight_decay=4e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)
    log = open(out / 'train.log', 'a')
    for epoch in range(1, args.epochs + 1):
        sampler.set_epoch(epoch)
        model.train(True)
        started, total = time.time(), 0.0
        for step, batch in enumerate(loader, 1):
            optimizer.zero_grad(set_to_none=True)
            result = model.train_local_batch(batch['images'].to(device, non_blocking=True),
                                             batch['attributes'].to(device, non_blocking=True))
            result['loss'].backward()
            optimizer.step()
            total += float(result['loss'].detach())
            if step % 200 == 0 or step == len(loader):
                line = (f'epoch {epoch} batch {step}/{len(loader)} loss {total / step:.5f} '
                        f'lr {optimizer.param_groups[0]["lr"]:.3g} {time.time() - started:.0f} s')
                print(line, flush=True); log.write(line + '\n'); log.flush()
        scheduler.step()
        target = out / f'epoch_{epoch:03d}.pth'
        torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, target)
        print(f'saved {target}', flush=True); log.write(f'saved {target}\n'); log.flush()


def prepare(args):
    """LODO data root for the original code, which requires every training query to have an image:
    the training queries are reduced to those present and the ids remapped (validation already
    keeps only present queries); image folders are linked to the full data set."""
    sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts'))
    import shutil
    from attrivision_lodo import write_split
    from upar.data import DOMAINS, load_split
    source, out = Path(args.source), Path(args.out)
    if out.exists():
        raise SystemExit(f'{out} exists; remove it first')
    names, labels, queries, ids = load_split(source, 'train')
    # Each image's query is its own label vector; keeping only distinct, present vectors also
    # merges duplicates (a train + validation root lists the validation queries after the train ones).
    vectors, ids = np.unique(np.asarray(queries)[ids], axis=0, return_inverse=True)
    ids = ids.reshape(-1)
    present = np.arange(len(vectors))
    queries = vectors
    write_split(out / 'annotations' / 'task2' / 'train', names, labels, queries, ids)
    shutil.copytree(source / 'annotations' / 'task2' / 'val', out / 'annotations' / 'task2' / 'val')
    for domain in DOMAINS:
        (out / domain).symlink_to((Path(args.data_dir) / domain).resolve())
    print(f'{out}: {len(names)} training images, {len(present)} distinct training queries')


def export(args):
    sys.path.insert(0, str(ROOT))
    from scripts.irra_member import PREPROCESS, convert
    from upar.results import load_predictions, save_predictions
    from upar.retrieval import predict_ensemble
    reference = load_predictions(args.reference)
    names = reference['image_names'].tolist()
    paths = [str(Path(args.data_dir).resolve() / n) for n in names]
    member, checkpoint = convert(args.checkpoint, text=bool(args.globals))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    member = member.to(device).eval()
    result = predict_ensemble([(member, PREPROCESS)], paths, device, 32, 8, flip=args.flip,
                              amp=device.type == 'cuda', globals_of=(0,) if args.globals else ())
    probabilities = result[0] if args.globals else result
    domain = names[0].split('/')[0]
    if args.out:
        save_predictions(args.out, probabilities, reference['labels'], reference['queries'], reference['ids'],
                         np.asarray(names), {'source': str(Path(args.checkpoint).resolve()), 'domain': domain})
        print(f'wrote {args.out}')
    if args.globals:
        # Image embeddings of the held-out gallery and text embeddings of every cached query
        # (canonical attribute order), for the image-text score in rescore_lodo.py.
        text = member.query_features(reference['queries'].astype(int), bytes(checkpoint['irra_bpe'].tolist()))
        np.savez(args.globals, image=result[2][0].astype(np.float16), text=text.cpu().numpy().astype(np.float16),
                 queries=reference['queries'].astype(np.uint8), image_names=np.asarray(names))
        print(f'wrote {args.globals}')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    s = sub.add_parser('stage2')
    s.add_argument('--repo', default='outputs/third_party/irra',
                   help='Working copy of third_party/irra made by scripts/irra_lodo.sh setup')
    s.add_argument('--stage1', required=True, help='Stage-1 epoch_002.pth (plain state_dict)')
    s.add_argument('--data', required=True, help='LODO data root (official layout, held-out domain removed)')
    s.add_argument('--out', required=True)
    s.add_argument('--epochs', type=int, default=2)
    s.add_argument('--workers', type=int, default=8)
    s.add_argument('--strip-heads', action='store_true',
                   help='--stage1 is a stage-2 checkpoint: drop its multiscale/local branch and train them anew')
    p = sub.add_parser('prepare')
    p.add_argument('--source', required=True, help='Our LODO data root (outputs/k8s/attrivision/data-DOMAIN)')
    p.add_argument('--out', required=True)
    p.add_argument('--data-dir', default='data')
    e = sub.add_parser('export')
    e.add_argument('--checkpoint', required=True)
    e.add_argument('--reference', required=True, help='Any held-out cache of the same domain (gallery order)')
    e.add_argument('--out', help='Held-out attribute-probability cache')
    e.add_argument('--globals', help='Also write image/text embeddings (.npz) for the image-text score')
    e.add_argument('--data-dir', default='data')
    e.add_argument('--flip', action='store_true', help='Horizontal-flip TTA')
    args = parser.parse_args()
    {'stage2': stage2, 'prepare': prepare, 'export': export}[args.command](args)


if __name__ == '__main__':
    main()
