import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

from upar.attributes import ATTRIBUTES
from upar.inference import InferenceEngine
from upar.model import build_model
from upar.retrieval import predict_ensemble, predict_probabilities, smooth_distances

PREPROCESS = {'height': 64, 'width': 32, 'mean': [0.5] * 3, 'std': [0.5] * 3}


def save_checkpoint(path, seed, half=False):
    torch.manual_seed(seed)
    model = build_model('clip_vit_tiny', image_size=(64, 32))
    state = {k: (v.half() if half and v.is_floating_point() else v) for k, v in model.state_dict().items()}
    torch.save({'format_version': 1, 'architecture': 'clip_vit_tiny', 'attribute_names': list(ATTRIBUTES),
                'preprocess': PREPROCESS, 'model_state': state, 'inference_batch_size': 2,
                'query_chunk_size': 4, 'training_config': {'retrieval': 'bce', 'preprocess': PREPROCESS}}, path)
    return model.eval()


class InferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        cls.folder = tempfile.TemporaryDirectory()
        root = Path(cls.folder.name)
        rng = np.random.default_rng(0)
        cls.images = []
        for i in range(3):
            path = root / f'{i}.png'
            Image.fromarray(rng.integers(0, 255, (90, 40, 3), dtype=np.uint8)).save(path)
            cls.images.append(str(path))
        cls.models = [save_checkpoint(root / f'model_{i}.pt', seed=i) for i in range(2)]
        (root / 'ensemble.json').write_text(json.dumps({'models': ['model_0.pt', 'model_1.pt'], 'tta_flip': True}))
        cls.root = root

    @classmethod
    def tearDownClass(cls):
        cls.folder.cleanup()

    def single(self, model, flip):
        return predict_probabilities(model, self.images, PREPROCESS, 'cpu', 2, 0, flip=flip)

    def test_int8_weights_round_trip(self):
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from build_submission import quantize_int8
        weight = torch.randn(300, 400)
        state = {'big.weight': weight.half(), 'small.weight': torch.randn(4, 4).half(), 'pos': torch.randn(1, 9, 8).half()}
        quantized, scales = quantize_int8(state)
        self.assertEqual(set(scales), {'big.weight'})
        self.assertEqual(quantized['big.weight'].dtype, torch.int8)
        scale = scales['big.weight'].float()
        self.assertEqual(tuple(scale.shape), (300, 400 // 64 if 400 % 64 == 0 else 1, 1))
        restored = (quantized['big.weight'].float().reshape(300, scale.shape[1], -1) * scale).reshape(300, 400)
        self.assertLess((restored - weight).abs().max().item(), weight.abs().max().item() / 127)
        grouped, group_scales = quantize_int8({'w.weight': torch.randn(256, 512).half()})
        self.assertEqual(tuple(group_scales['w.weight'].shape), (256, 8, 1))
        self.assertTrue(torch.equal(quantized['small.weight'], state['small.weight']))

    def test_ensemble_averages_members(self):
        engine = InferenceEngine([self.root / 'model_0.pt', self.root / 'model_1.pt'], 'cpu')
        expected = (self.single(self.models[0], False) + self.single(self.models[1], False)) / 2
        np.testing.assert_allclose(engine.predict(self.images), expected, rtol=1e-5, atol=1e-6)
        self.assertEqual(engine.retrieval, 'bce')

    def test_flip_tta_averages_flipped_view(self):
        from upar.data import ImageDataset, make_transform
        images = torch.stack([ImageDataset(self.images, make_transform(PREPROCESS))[i] for i in range(3)])
        with torch.no_grad():
            manual = (self.models[0](images).sigmoid() + self.models[0](images.flip(3)).sigmoid()) / 2
        np.testing.assert_allclose(self.single(self.models[0], True), manual.numpy(), rtol=1e-5, atol=1e-6)

    def test_assets_manifest_enables_ensemble_and_tta(self):
        engine = InferenceEngine(self.root, 'cpu')
        self.assertEqual(len(engine.models), 2)
        self.assertTrue(engine.tta_flip)
        expected = (self.single(self.models[0], True) + self.single(self.models[1], True)) / 2
        np.testing.assert_allclose(engine.predict(self.images), expected, rtol=1e-5, atol=1e-6)

    def test_ensemble_with_different_normalizations(self):
        other = {**PREPROCESS, 'mean': [0.48, 0.46, 0.41], 'std': [0.27, 0.26, 0.28]}
        members = [(self.models[0], PREPROCESS), (self.models[1], other)]
        expected = (self.single(self.models[0], True)
                    + predict_probabilities(self.models[1], self.images, other, 'cpu', 2, 0, flip=True)) / 2
        np.testing.assert_allclose(predict_ensemble(members, self.images, 'cpu', 2, 0, flip=True), expected,
                                   rtol=1e-5, atol=1e-6)

    def test_per_member_flip(self):
        members = [(m, PREPROCESS) for m in self.models]
        mixed = predict_ensemble(members, self.images, 'cpu', flip=[True, False])
        first = predict_probabilities(self.models[0], self.images, PREPROCESS, 'cpu', flip=True)
        second = predict_probabilities(self.models[1], self.images, PREPROCESS, 'cpu', flip=False)
        np.testing.assert_allclose(mixed, (first + second) / 2, rtol=1e-5, atol=1e-6)
        with self.assertRaises(ValueError):
            predict_ensemble(members, self.images, 'cpu', flip=[True])

    def test_float16_weights_load_for_float32_inference(self):
        path = self.root / 'half.pt'
        model = save_checkpoint(path, seed=0, half=True)
        engine = InferenceEngine(path, 'cpu')
        self.assertTrue(all(p.dtype == torch.float32 for p in engine.model.parameters()))
        np.testing.assert_allclose(engine.predict(self.images), self.single(model, False), atol=2e-3)


class GallerySmoothingTests(unittest.TestCase):
    def test_matches_neighbour_definition(self):
        rng = np.random.default_rng(0)
        images = rng.normal(size=(300, 8)).astype(np.float32)
        distances = rng.normal(size=(7, 300)).astype(np.float32)
        unit = images / np.linalg.norm(images, axis=1, keepdims=True)
        similarity = unit @ unit.T
        np.fill_diagonal(similarity, -np.inf)
        near = np.argsort(-similarity, axis=1)[:, :5]
        d = distances.astype(np.float64)
        expected = 0.7 * d + 0.3 * d[:, near].mean(-1)
        np.testing.assert_allclose(smooth_distances(distances.copy(), images, 5, 0.3, chunk_size=64, query_chunk=3),
                                   expected, atol=1e-5)
        mutual = (near[near] == np.arange(len(near))[:, None, None]).any(-1)
        expected = d + 0.3 / 5 * ((d[:, near] - d[:, :, None]) * mutual).sum(-1)
        np.testing.assert_allclose(smooth_distances(distances.copy(), images, 5, 0.3, mutual=True), expected, atol=1e-5)
        with self.assertRaises(ValueError):
            smooth_distances(distances.copy(), images, 300, 0.3)


if __name__ == '__main__':
    unittest.main()
