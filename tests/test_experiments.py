import json
from pathlib import Path
import unittest

import numpy as np
import torch

from upar.config import model_options, normalize_config
from upar.data import make_transform, select_domains
from upar.vit import openai_visual_state, resize_positional_embedding
from upar.experiments import DeepMARLoss, ModelEMA, smooth_targets
from upar.model import build_model
from upar.retrieval import distance_matrix

CONFIGS = Path(__file__).resolve().parents[1] / 'configs'


class ExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)

    def test_config_defaults(self):
        original = {'backbone': 'clip_vit_l14', 'loss': 'bce'}
        config = normalize_config(original)
        self.assertNotIn('ema', original)
        self.assertEqual((config['backbone'], config['architecture']), ('clip_vit_l14', 'clip_vit_l14'))
        self.assertFalse(config['ema'])
        self.assertEqual(config['label_smoothing'], 0)
        self.assertEqual(config['retrieval'], 'l1')
        self.assertEqual(model_options(config)['image_size'], (256, 128))

    def test_submitted_configs_load(self):
        for path in sorted(CONFIGS.glob('*.json')):
            with self.subTest(config=path.name):
                config = normalize_config(json.loads(path.read_text()))
                self.assertIn(config['backbone'], ('clip_vit_l14', 'dinov2_vit_l14', 'pe_core_vit_l14'))
                self.assertEqual(config['loss'], 'deepmar_bce')

    def test_dropout_keeps_state_dict_layout(self):
        torch.manual_seed(0)
        plain = build_model('clip_vit_l14', image_size=(28, 14))
        dropped = build_model('clip_vit_l14', image_size=(28, 14), dropout=0.7)
        self.assertEqual(list(plain.state_dict()), list(dropped.state_dict()))
        dropped.load_state_dict(plain.state_dict(), strict=True)
        images = torch.randn(2, 3, 28, 14)
        plain.eval(); dropped.eval()
        with torch.no_grad():
            torch.testing.assert_close(plain(images), dropped(images))
            dropped.train()
            self.assertFalse(torch.allclose(dropped(images), dropped(images)))

    def test_deepmar_weights(self):
        rate = torch.tensor([0.1, 0.5])
        logits = torch.zeros(1, 2)
        targets = torch.tensor([[1.0, 0.0]])
        expected = torch.log(torch.tensor(2.0)) * torch.exp(torch.tensor([0.9, 0.5])).mean()
        torch.testing.assert_close(DeepMARLoss(rate)(logits, targets), expected)
        # Weights follow hard labels when targets are smoothed.
        smoothed = smooth_targets(targets, 0.1)
        weights = torch.exp(torch.tensor([0.9, 0.5]))
        manual = torch.nn.functional.binary_cross_entropy_with_logits(logits, smoothed, weight=weights)
        torch.testing.assert_close(DeepMARLoss(rate)(logits, smoothed), manual)

    def test_augmix_training_transform_only(self):
        from PIL import Image
        preprocess = {'height': 64, 'width': 32, 'mean': [0.5] * 3, 'std': [0.5] * 3}
        image = Image.new('RGB', (40, 90), (120, 30, 200))
        self.assertEqual(make_transform(preprocess, True, augmix=True)(image).shape, (3, 64, 32))
        self.assertNotIn('AugMix', repr(make_transform(preprocess, False, augmix=True)))
        self.assertNotIn('AugMix', repr(make_transform(preprocess, True)))

    def test_clip_vit_shapes_head_and_openai_keys(self):
        model = build_model('clip_vit_l14', image_size=(28, 14), dropout=0.5)
        self.assertEqual(model.grid, (2, 1))
        self.assertEqual(model.head_prefix, 'head.')
        self.assertEqual(type(model.transformer.resblocks[0].mlp.gelu).__name__, 'QuickGELU')
        model.eval()
        with torch.no_grad():
            self.assertEqual(tuple(model(torch.randn(2, 3, 28, 14)).shape), (2, 40))
        openai = {'visual.' + k: v for k, v in model.state_dict().items() if not k.startswith('head.')}
        openai['visual.positional_embedding'] = torch.randn(1 + 16 * 16, 1024)
        openai.update({'visual.proj': torch.zeros(1024, 768), 'token_embedding.weight': torch.zeros(3, 768)})
        visual = openai_visual_state(openai, model.grid)
        self.assertEqual(sorted(visual), sorted(k for k in model.state_dict() if not k.startswith('head.')))
        self.assertEqual(tuple(visual['positional_embedding'].shape), (1 + 2, 1024))
        square = torch.randn(1 + 9, 8)
        torch.testing.assert_close(resize_positional_embedding(square, (3, 3)), square)
        with self.assertRaises(ValueError):
            build_model('clip_vit_l14', image_size=(30, 14))

    def test_holdout_domain_config_and_selection(self):
        base = {'backbone': 'clip_vit_l14'}
        self.assertIsNone(normalize_config(base)['holdout_domain'])
        self.assertEqual(normalize_config({**base, 'holdout_domain': 'PETA'})['holdout_domain'], 'PETA')
        with self.assertRaises(ValueError):
            normalize_config({**base, 'holdout_domain': 'RAP2'})
        self.assertEqual(normalize_config(base)['linear_probe_epochs'], 0)
        for bad in ({'linear_probe_epochs': -1}, {'linear_probe_epochs': 1.5}, {'linear_probe_lr': 0}):
            with self.assertRaises(ValueError):
                normalize_config({**base, **bad})
        paths = ['Market1501/a.jpg', 'PETA/b.png', 'PA100k/c.jpg', 'PETA/d.png']
        labels, ids = np.arange(4)[:, None] * np.ones((1, 40)), np.array([3, 1, 2, 0])
        kept, kept_labels, kept_ids = select_domains(paths, labels, ids, ['PETA', 'Market1501'])
        self.assertEqual(kept, ['Market1501/a.jpg', 'PETA/b.png', 'PETA/d.png'])
        self.assertEqual(kept_labels[:, 0].tolist(), [0, 1, 3])
        self.assertEqual(kept_ids.tolist(), [3, 1, 0])
        with self.assertRaises(ValueError):
            select_domains(paths, labels, ids, ['RAP2'])

    def test_invalid_switches_and_alias_conflicts_fail(self):
        for changed in ({'ema': 'false'}, {'label_smoothing': 1}, {'retrieval': 'unknown'}, {'backbone': 'convnext_tiny'},
                        {'backbone': 'clip_vit_l14', 'architecture': 'dinov2_vit_l14'}):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                normalize_config({'backbone': 'clip_vit_l14', **changed})

    def test_binary_smoothing_and_zero_identity(self):
        labels = torch.tensor([[0., 1.]])
        self.assertIs(smooth_targets(labels, 0), labels)
        torch.testing.assert_close(smooth_targets(labels, 0.2), torch.tensor([[0.1, 0.9]]))

    def test_ema_updates_shadow_without_changing_live_model(self):
        model = torch.nn.Linear(2, 1, bias=False)
        with torch.no_grad(): model.weight.fill_(2)
        ema = ModelEMA(model, 0.75)
        with torch.no_grad(): model.weight.fill_(6)
        ema.update(model)
        torch.testing.assert_close(ema.model.weight, torch.full((1, 2), 3.))
        torch.testing.assert_close(model.weight, torch.full((1, 2), 6.))
        self.assertEqual(ema.num_updates, 1)
        self.assertFalse(ema.model.training)
        self.assertFalse(ema.model.weight.requires_grad)

    def test_category_query_head_starts_neutral_and_trains_in_probe(self):
        from upar.vit import CATEGORIES
        self.assertEqual(sorted(i for c in CATEGORIES for i in c), list(range(40)))
        torch.manual_seed(0)
        plain = build_model('clip_vit_l14', image_size=(28, 14)).eval()
        query = build_model('clip_vit_l14', image_size=(28, 14), query_head=True).eval()
        missing, unexpected = query.load_state_dict(plain.state_dict(), strict=False)
        self.assertFalse(unexpected); self.assertTrue(all(k.startswith('head_query.') for k in missing))
        x = torch.randn(2, 3, 28, 14)
        with torch.no_grad():
            torch.testing.assert_close(query(x), plain(x))  # zero-initialized output layers
        trainable = [n for n, _ in query.named_parameters() if n.startswith(query.head_prefix)]
        self.assertTrue(any(n.startswith('head_query.') for n in trainable) and 'head.weight' in trainable)
        self.assertFalse(any(n.startswith('transformer.') for n in trainable))
        self.assertTrue(normalize_config({'backbone': 'clip_vit_l14', 'query_head': True})['query_head'])
        with torch.no_grad():
            query.head_query.out[4].bias.fill_(1.0)  # upper-body colour -> attributes 8..19
            delta = query(x) - plain(x)
        expected = torch.zeros_like(delta); expected[:, 8:20] = 1
        torch.testing.assert_close(delta, expected, atol=1e-5, rtol=0)

    def test_submission_modules_import_only_packaged_modules(self):
        # Regression: config.py once imported clip_text at module level, which submissions do
        # not ship, so the server failed with ModuleNotFoundError (10-15).
        import ast
        from build_submission import INFERENCE_MODULES
        root = Path(__file__).resolve().parents[1] / 'upar'
        for name in INFERENCE_MODULES:
            tree = ast.parse((root / name).read_text())
            for node in tree.body:  # module level only; function-level imports are lazy
                if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module:
                    self.assertIn(node.module.split('.')[0] + '.py', INFERENCE_MODULES, f'{name} imports {node.module}')
        from upar.clip_text import TEXT_SPECS
        from upar.config import TEXT_PRIOR_SOURCES
        self.assertEqual(set(TEXT_PRIOR_SOURCES), set(TEXT_SPECS))

    def test_text_prior_probing_needs_query_head(self):
        base = json.loads((CONFIGS / 'metaclip_b7_val.json').read_text())
        self.assertTrue(normalize_config({**base, 'query_head': True, 'linear_probe_epochs': 1})['query_head'])
        with self.assertRaises(ValueError):
            normalize_config({**base, 'linear_probe_epochs': 1})

    def test_dinov2_variant(self):
        from upar.pretrained import SOURCES
        model = build_model('dinov2_vit_l14', image_size=(28, 14), dropout=0.5)
        x = torch.randn(2, 3, 28, 14)
        with torch.no_grad():
            self.assertEqual(tuple(model(x).shape), (2, 40))
        self.assertEqual(tuple(model.pos_embed.shape), (1, 3, 1024))
        self.assertEqual(SOURCES['dinov2_l14']['architecture'], 'dinov2_vit_l14')
        config = normalize_config(json.loads((CONFIGS / 'dinov2.json').read_text()))
        self.assertEqual(config['preprocess']['mean'], [0.485, 0.456, 0.406])

    def test_pe_core_variant(self):
        from upar.pretrained import SOURCES
        from upar.vit import pe_rope, timm_pe_state
        model = build_model('pe_core_vit_l14', image_size=(28, 14), dropout=0.5).eval()
        x = torch.randn(2, 3, 28, 14)
        with torch.no_grad():
            self.assertEqual(tuple(model(x).shape), (2, 40))
            self.assertEqual(tuple(model.tokens(x).shape), (2, 3, 1024))
        sin, cos = pe_rope((2, 1), 64)
        self.assertEqual(tuple(sin.shape), (2, 64))
        state = timm_pe_state({'pos_embed': torch.randn(1, 577, 1024), 'head.weight': torch.zeros(1)}, model.grid)
        self.assertEqual(tuple(state['pos_embed'].shape), (1, 3, 1024))
        self.assertNotIn('head.weight', state)
        self.assertEqual(SOURCES['pe_core_l14']['architecture'], 'pe_core_vit_l14')
        config = normalize_config(json.loads((CONFIGS / 'pe_core_val.json').read_text()))
        self.assertEqual((config['preprocess']['height'], config['preprocess']['mean']), (252, [0.5, 0.5, 0.5]))

    def test_irra_member_forward(self):
        model = build_model('irra_clip_l14', image_size=(28, 14)).eval()
        with torch.no_grad():
            self.assertEqual(tuple(model(torch.randn(2, 3, 28, 14)).shape), (2, 40))

    def test_e1_member_forward(self):
        model = build_model('e1_clip_l14', image_size=(224, 224))
        groups = [[0, 1, 2, -1], [3, ~3]] + [[a, ~a] for a in range(4, 40) if a not in (0, 1, 2, 3)][:10]
        state = 0
        for category, group in enumerate(groups):
            for attribute in group:
                if state < 52:
                    model.state_category[state] = category
                    model.state_attribute[state] = attribute if attribute >= 0 else -1
                state += 1
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            probabilities = model(x).sigmoid()
            fused = model.state_log_probabilities(x).exp()
        self.assertEqual(tuple(probabilities.shape), (1, 40))
        torch.testing.assert_close(probabilities[0, 3], fused[0, 4], rtol=1e-4, atol=1e-6)  # Gender-Female state

    def test_train_py_training_transform_normalizes_like_inference(self):
        # Regression: a positional argument once landed in `normalize`, so training inputs were
        # not normalized while inference inputs were (10-13 to 10-14).
        import ast
        tree = ast.parse(Path(__file__).resolve().parents[1].joinpath('train.py').read_text())
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, 'id', None) == 'make_transform']
        training = [c for c in calls if len(c.args) >= 2 and isinstance(c.args[1], ast.Constant) and c.args[1].value is True]
        self.assertTrue(training)
        for call in training:
            self.assertLessEqual(len(call.args), 3, 'pass normalize by keyword')
            self.assertFalse(any(k.arg == 'normalize' for k in call.keywords))

    def test_bce_retrieval_matches_direct_log_likelihood(self):
        rng = np.random.default_rng(7)
        queries = rng.integers(0, 2, (3, 40))
        probabilities = rng.random((5, 40), dtype=np.float32)
        p = np.clip(probabilities, 1e-6, 1 - 1e-6)
        expected = -(queries[:, None, :] * np.log(p)[None, :, :] +
                     (1 - queries[:, None, :]) * np.log1p(-p)[None, :, :]).sum(2)
        np.testing.assert_allclose(distance_matrix(queries, probabilities, 2, 'bce'), expected, atol=1e-5)
        self.assertTrue(np.isfinite(distance_matrix(queries, np.round(probabilities), 2, 'bce')).all())


if __name__ == '__main__':
    unittest.main()
