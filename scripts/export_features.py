"""Pooled image features of a LODO soup on its held-out gallery (the input of the attribute head),
for swap_lodo.py --neighbour-cache.

The soup is rebuilt from the metadata of a soup_lodo.py cache (members, base, alpha), so the
features belong to exactly the model in the ensemble; its probabilities are compared with the
cache as a check. A forward pre-hook on `model.head` collects the features (the head dropout is
off in eval mode), so the model code is unchanged.

    python scripts/export_features.py --soup-cache "outputs/soups/wise-dinov2-e4/{domain}/alpha0.8.npz" \
        --domains PETA --output "outputs/features/dinov2-e4-wise0.8/{domain}.npz"
"""
import argparse
import contextlib
import io
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

from soup_lodo import run_path  # noqa: E402
from upar.results import load_predictions  # noqa: E402
from upar.retrieval import predict_probabilities  # noqa: E402
from upar.soup import load_checkpoint, mix_checkpoints, model_from_checkpoint  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--soup-cache', required=True, help='soup_lodo.py cache pattern with {domain}')
    parser.add_argument('--domains', nargs='+', default=['Market1501', 'PETA'])
    parser.add_argument('--output', required=True, help='Feature file pattern with {domain}')
    parser.add_argument('--data-dir', default='data')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    data_root = Path(args.data_dir).resolve()
    for domain in args.domains:
        path = ROOT / args.soup_cache.format(domain=domain)
        cache = load_predictions(path)
        with np.load(path) as archive:
            metadata = json.loads(str(archive['metadata']))
        members, base, alpha = metadata['members'], metadata.get('base') or [], float(metadata['alpha'])
        specs = base + members
        checkpoints = [load_checkpoint(run_path(s, domain)[0] / f'{run_path(s, domain)[1]}.pt') for s in specs]
        if base:
            weights = [(1 - alpha) / len(base)] * len(base) + [alpha / len(members)] * len(members)
        else:
            weights = [1 / len(members)] * len(members)
        mixed = mix_checkpoints(checkpoints, weights, specs)
        model = model_from_checkpoint(mixed, args.device)
        features = []
        handle = model.head.register_forward_pre_hook(lambda module, inputs: features.append(inputs[0].float().cpu()))
        names = cache['image_names'].tolist()
        with contextlib.redirect_stdout(io.StringIO()):
            probabilities = predict_probabilities(model, [str(data_root / n) for n in names], mixed['preprocess'],
                                                  args.device, args.batch_size, args.workers)
        handle.remove()
        image = torch.cat(features).numpy()
        if len(image) != len(names):
            sys.exit(f'{domain}: {len(image)} features for {len(names)} images')
        difference = float(np.abs(probabilities - cache['probs']).max())
        target = ROOT / args.output.format(domain=domain)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez(target, image_names=cache['image_names'], image=image.astype(np.float16),
                 metadata=np.asarray(json.dumps({'soup_cache': str(path), **metadata})))
        print(f'{domain}: {image.shape} features -> {target}; max probability difference to the cache '
              f'{difference:.2e}', flush=True)
        del model


if __name__ == '__main__':
    main()
