from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

from upar.attributes import ATTRIBUTES
from upar.model import build_model
from upar.results import load_predictions, save_predictions
from upar.soup import mix_checkpoints, wise_weights

ROOT = Path(__file__).resolve().parents[1]
PREPROCESS = {'height': 64, 'width': 32, 'mean': [0.5] * 3, 'std': [0.5] * 3}


def checkpoint(seed):
    torch.manual_seed(seed)
    model = build_model('clip_vit_tiny', image_size=(64, 32))
    return {'format_version': 1, 'architecture': 'clip_vit_tiny', 'attribute_names': list(ATTRIBUTES),
            'preprocess': PREPROCESS, 'model_state': model.state_dict(), 'inference_batch_size': 2,
            'query_chunk_size': 4, 'epoch': seed, 'validation_mADM': 0.0,
            'training_config': {'retrieval': 'bce', 'holdout_domain': 'PETA', 'preprocess': PREPROCESS}}


class SoupTests(unittest.TestCase):
    def test_weighted_mix(self):
        a, b, c = checkpoint(0), checkpoint(1), checkpoint(2)
        mixed = mix_checkpoints([a, b, c], wise_weights(0.6, 2))
        for key, value in a['model_state'].items():
            if value.is_floating_point():
                expected = 0.4 * value + 0.3 * b['model_state'][key] + 0.3 * c['model_state'][key]
                torch.testing.assert_close(mixed['model_state'][key], expected)
        base_only = mix_checkpoints([a, b], wise_weights(0.0, 1))
        for key, value in a['model_state'].items():
            torch.testing.assert_close(base_only['model_state'][key], value)

    def test_rejects_mismatch(self):
        a, b = checkpoint(0), checkpoint(1)
        b['preprocess'] = {**PREPROCESS, 'height': 128}
        with self.assertRaises(ValueError):
            mix_checkpoints([a, b], [0.5, 0.5])
        with self.assertRaises(ValueError):
            mix_checkpoints([a, a], [0.5, 0.6])

    def test_soup_lodo_script(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            rng = np.random.default_rng(0)
            names = [f'PETA/{i}.png' for i in range(4)]
            for name in names:
                (root / 'data' / name).parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(rng.integers(0, 255, (90, 40, 3), dtype=np.uint8)).save(root / 'data' / name)
            queries = np.eye(40, dtype=np.uint8)[:2]
            ids = np.array([0, 1, 0, 1])
            for run, seed in (('a', 0), ('b', 1)):
                run_dir = root / f'{run}-PETA'
                run_dir.mkdir()
                torch.save(checkpoint(seed), run_dir / 'epoch1.pt')
                torch.save(checkpoint(seed + 10), run_dir / 'epoch4.pt')
                save_predictions(run_dir / 'holdout_predictions_epoch4.npz', np.full((4, 40), 0.5, np.float32),
                                 queries[ids], queries, ids, names, {})
            result = subprocess.run(
                [sys.executable, 'scripts/soup_lodo.py', f'{root}/a-{{domain}}:4', f'{root}/b-{{domain}}:4',
                 '--base', f'{root}/a-{{domain}}:1', '--alphas', '0', '0.5', '1', '--domains', 'PETA',
                 '--data-dir', str(root / 'data'), '--output-dir', str(root / 'out'), '--device', 'cpu',
                 '--batch-size', '2', '--workers', '0'],
                cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('mean mADM', result.stdout)
            cache = load_predictions(root / 'out/PETA/alpha0.5.npz')
            self.assertEqual(cache['probs'].shape, (4, 40))


if __name__ == '__main__':
    unittest.main()
