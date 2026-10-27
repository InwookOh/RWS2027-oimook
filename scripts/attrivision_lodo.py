"""Runs of a teammate's E1 model (third_party/attrivision), scored like ours. Its code is used
unchanged; this script only

1. `prepare`: writes a data root whose train split excludes the held-out domain and whose val
   split is our LODO held-out gallery (val images of that domain, full query list), with links
   to the image folders (`--domain full`: all domains, for submissions);
2. `e1a`, `e1b`, `export-e1`: E1-A (CLIP ViT-L/14 image and text towers trained against 52
   category-state prompts, stopped after a few epochs of its 100-epoch schedule), E1-B (a
   frozen-encoder AttriQuery local head) and their held-out caches (the score fuses global and
   local category log-probabilities, alpha 0.5), so scripts/ensemble_lodo.py can mix E1 with our
   members.

    python scripts/attrivision_lodo.py prepare --domain PETA --out outputs/k8s/attrivision/data-PETA
    python scripts/attrivision_lodo.py e1a --data outputs/k8s/attrivision/data-PETA --out outputs/k8s/e1/lodo-PETA/a --stop-after 2
    python scripts/attrivision_lodo.py e1b --data outputs/k8s/attrivision/data-PETA \
        --init outputs/k8s/e1/lodo-PETA/a/checkpoint_epoch_002.pth --out outputs/k8s/e1/lodo-PETA/b6 --stop-after 3
    python scripts/attrivision_lodo.py export-e1 --data outputs/k8s/attrivision/data-PETA \
        --out outputs/k8s/e1/lodo-PETA/b6 --stage b --epochs 3
"""
import argparse
import csv
import json
import shlex
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from upar.attributes import ATTRIBUTES  # noqa: E402
from upar.data import DOMAINS, load_split, select_domains  # noqa: E402
from upar.results import save_predictions  # noqa: E402


def write_split(folder, names, labels, queries, ids):
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / 'gt.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['# image', *ATTRIBUTES])
        writer.writerows([name, *map(int, row)] for name, row in zip(names, labels))
    with (folder / 'queries.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['# ' + ATTRIBUTES[0], *ATTRIBUTES[1:]])
        writer.writerows([*map(int, row)] for row in queries)
    with (folder / 'ids.csv').open('w', newline='') as f:
        csv.writer(f).writerows([name, int(i)] for name, i in zip(names, ids))


def prepare(args):
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f'{out} exists; remove it first')
    train_names, train_labels, train_queries, train_ids = load_split(args.data_dir, 'train')
    val_names, val_labels, val_queries, val_ids = load_split(args.data_dir, 'val')
    seen = [d for d in DOMAINS if d != args.domain]
    names, labels, ids = select_domains(train_names, train_labels, train_ids, seen)
    queries = train_queries
    if args.with_val:  # also train on the seen-domain validation images (their queries appended)
        extra_names, extra_labels, extra_ids = select_domains(val_names, val_labels, val_ids, seen)
        names, labels = names + extra_names, np.concatenate([labels, extra_labels])
        ids = np.concatenate([ids, extra_ids + len(train_queries)])
        queries = np.concatenate([train_queries, val_queries])
    write_split(out / 'annotations' / 'task2' / 'train', names, labels, queries, ids)
    # Validation: the held-out domain, or with --domain full the whole official split (in-sample
    # with --with-val, only logged; epochs are fixed by --stop-after).
    names, labels, ids = select_domains(val_names, val_labels, val_ids, [args.domain] if args.domain != 'full' else DOMAINS)
    # The A7 evaluator needs every query to match some gallery image: keep only the queries
    # present in the held-out gallery (our caches keep the full list; see export).
    present, ids = np.unique(ids, return_inverse=True)
    write_split(out / 'annotations' / 'task2' / 'val', names, labels, val_queries[present], ids)
    for domain in DOMAINS:
        (out / domain).symlink_to(Path(args.data_dir).resolve() / domain)
    (out / 'lodo.json').write_text(json.dumps({'domain': args.domain}))
    print(f'{out}: train = {seen}{" + validation" if args.with_val else ""}, val = {args.domain} ({len(names)} images)')


def import_repo(repo):
    """Put the A7 code first on the path. It ships its own `upar` package; ours is dropped from
    sys.modules (the functions this script already imported keep working)."""
    for name in [m for m in sys.modules if m == 'upar' or m.startswith('upar.')]:
        del sys.modules[name]
    source = Path(repo).expanduser().resolve() / 'examples' / 'task2'
    for path in (source / 'attrivision_a7', source / 'sample_code_submission'):
        sys.path.insert(0, str(path))


def _e1_runner(repo):
    return Path(repo).expanduser().resolve() / 'examples' / 'task2' / 'sample_code_submission' / 'attrivision' / 'shared_clip_l14' / 'run.py'


def e1a(args):
    """E1-A: A7 on CLIP L/14 with its 100-epoch schedule, stopped after --stop-after epochs."""
    import subprocess
    out = Path(args.out).resolve()
    command = [sys.executable, str(_e1_runner(args.repo)), 'a7', '--mode', 'train', '--data-root',
               str(Path(args.data).resolve()), '--device', 'cuda:0', '--batch-size', str(args.batch_size),
               '--grad-cache-steps', str(128 // args.batch_size), '--eval-batch-size', '32', '--distill-weight', '0',
               '--stop-after-epoch', str(args.stop_after), '--num-workers', str(args.workers), '--output-dir', str(out),
               *shlex.split(args.extra)]
    print(' '.join(command), flush=True)
    subprocess.run(command, cwd=Path(args.repo).expanduser().resolve(), check=True)


def e1b(args):
    """E1-B0: frozen E1-A encoders, AttriQuery local head with the config's 30-epoch schedule;
    every epoch is kept as checkpoint_epoch_NNN.pth and the run is stopped after --stop-after."""
    import subprocess
    import time
    repo = Path(args.repo).expanduser().resolve()
    out = Path(args.out).resolve()
    config = repo / 'examples' / 'task2' / 'sample_code_submission' / 'attrivision' / 'configs' / 'e1_shared_clip_l14.json'
    command = [sys.executable, str(_e1_runner(repo)), 'train', '--stage', 'local', '--consistency-variant', 'B0',
               '--init-checkpoint', str(Path(args.init).resolve()), '--config', str(config),
               '--data-root', str(Path(args.data).resolve()), '--device', 'cuda:0', '--batch-size', str(args.batch_size),
               '--grad-cache-steps', str(128 // args.batch_size), '--eval-batch-size', '32',
               '--num-workers', str(args.workers), '--output-dir', str(out), *shlex.split(args.extra)]
    print(' '.join(command), flush=True)
    process = subprocess.Popen(command, cwd=repo)
    target = out / f'checkpoint_epoch_{args.stop_after:03d}.pth'
    while process.poll() is None:
        if target.exists():
            time.sleep(60)  # let the atomic save and its evaluation finish
            process.terminate()
            process.wait()
            print(f'[attrivision_lodo] stopped after epoch {args.stop_after}', flush=True)
            return
        time.sleep(30)
    if process.returncode:
        raise SystemExit(process.returncode)


def export_e1(args):
    """Held-out caches from E1 checkpoints: E1-A (global score only) or E1-B (global and local
    category log-probabilities fused with alpha 0.5, as in the E1 evaluator)."""
    import torch
    data = Path(args.data).resolve()
    domain = json.loads((data / 'lodo.json').read_text())['domain']
    val_names, val_labels, queries, val_ids = load_split(args.data_dir, 'val')
    names, labels, ids = select_domains(val_names, val_labels, val_ids, [domain])
    import_repo(args.repo)
    import attrivision.engine  # noqa: F401  (A7's own import order)
    from attrivision.attrquery_fromscratch_jointfce import encode_gallery_scores, fused_category_probabilities
    from attrivision.checkpoint import load_model
    from attrivision.compare_inference import encode_texts
    from attrivision.datasets.attribute_prompts import CategoryPromptMapper
    from attrivision.engine.evaluator_abpr import encode_gallery
    from attrivision.native52_retrieval_ablation import category_softmax
    from attrivision.shared_clip_l14.experiment import _load_shared_model
    from attrivision.transforms import build_eval_transform
    if names != load_split(data, 'val')[0]:
        raise ValueError(f'{data}: held-out gallery differs from {args.data_dir}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out = Path(args.out)
    transform = build_eval_transform(224, 'resize_pad_crop')
    for epoch in args.epochs:
        path = out / f'checkpoint_epoch_{epoch:03d}.pth'
        target = out / f'holdout_predictions_epoch{epoch}.npz'
        if args.stage == 'a':
            model, payload = load_model(path, device)
            mapper = CategoryPromptMapper(list(payload['attribute_names']))
            gallery = encode_gallery(model, names, [data], transform, device, args.batch_size, args.workers, True)
            states = encode_texts(model, mapper.prompts, device, True)
            groups = [np.asarray(g, dtype=np.int64) for g in mapper.category_indices()]
            state_probs = category_softmax((gallery.float() @ states.float().T).numpy() / args.temperature, groups)
        else:
            model, payload = _load_shared_model(path, device)
            mapper = CategoryPromptMapper(list(payload['attribute_names']))
            with torch.inference_mode():
                states = model.encode_text(model.base.tokenize(mapper.prompts).to(device)).float()
            global_scores, local_scores = encode_gallery_scores(model, names, [data], transform, states, device,
                                                                args.batch_size, args.workers, True)
            groups = [np.asarray(g, dtype=np.int64) for g in mapper.category_indices()]
            state_probs = fused_category_probabilities(global_scores, local_scores, groups, args.temperature, 0.5)[2]
        state_index = {key: i for i, key in enumerate(mapper.keys)}
        columns = {}
        for _, attributes, keys, _ in mapper._MULTI_GROUPS:
            columns.update(zip(attributes, keys))
        for attribute, positive, _ in mapper._BINARY_GROUPS:
            columns[attribute] = positive
        probs = np.stack([state_probs[:, state_index[columns[a]]] for a in ATTRIBUTES], axis=1)
        save_predictions(target, probs, labels, queries, ids, names,
                         {'epoch': epoch, 'model': f'attrivision_e1{args.stage}', 'temperature': args.temperature,
                          'weight_source': 'model'})
        print(f'[attrivision_lodo] wrote {target}', flush=True)
        del model
        torch.cuda.empty_cache()

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--domain', required=True, choices=(*DOMAINS, 'full'), help="Held-out domain, or 'full' (none)")
    p.add_argument('--out', required=True)
    p.add_argument('--data-dir', default='data')
    p.add_argument('--with-val', action='store_true', help='Train on train + seen-domain validation images')
    for name in ('e1a', 'e1b'):
        p = sub.add_parser(name)
        p.add_argument('--repo', default=str(ROOT / 'third_party' / 'attrivision'), help='E1 code (default: third_party/attrivision)')
        p.add_argument('--data', required=True)
        p.add_argument('--out', required=True)
        p.add_argument('--stop-after', type=int, required=True)
        p.add_argument('--batch-size', type=int, default=8, help='Microbatch; Gradient Cache keeps 128 per update')
        p.add_argument('--workers', type=int, default=8)
        p.add_argument('--extra', default='', help='Extra options passed to the E1 code')
        if name == 'e1b':
            p.add_argument('--init', required=True, help='E1-A checkpoint_epoch_NNN.pth')
    p = sub.add_parser('export-e1')
    p.add_argument('--repo', default=str(ROOT / 'third_party' / 'attrivision'), help='E1 code (default: third_party/attrivision)')
    p.add_argument('--data', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--stage', choices=('a', 'b'), required=True)
    p.add_argument('--epochs', type=int, nargs='+', required=True)
    p.add_argument('--temperature', type=float, default=0.01)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--data-dir', default='data')
    args = parser.parse_args()
    {'prepare': prepare, 'e1a': e1a, 'e1b': e1b,
     'export-e1': export_e1}[args.command](args)


if __name__ == '__main__':
    main()
