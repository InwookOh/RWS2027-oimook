import unittest

import numpy as np

from upar.attributes import ATTRIBUTES
from upar.metrics import evaluate_retrieval, query_metrics
from upar.retrieval import distance_matrix, l1_distances


class MetricTests(unittest.TestCase):
    def setUp(self):
        self.query = np.zeros(40, dtype=np.float32)
        self.exact = self.query.copy()
        self.near = self.query.copy(); self.near[0] = 1
        self.far = np.ones(40, dtype=np.float32)

    def test_official_attribute_count(self):
        self.assertEqual(len(ATTRIBUTES), 40)
        self.assertEqual(ATTRIBUTES[39], 'Accessory-Hat')

    def test_hand_calculated_partial_agreement(self):
        labels = np.stack([self.near, self.exact, self.far, self.exact])
        result = query_metrics(self.query, labels, [0, 1, 2, 3])
        # Exact matches at ranks 2 and 4; near match gets normalized credit 37/41.
        expected_adm = ((1 + 37 / 41) / 2 + (2 + 37 / 41) / 4) / 2
        self.assertAlmostEqual(result['mADM'], expected_adm)
        self.assertAlmostEqual(result['mAP'], 0.5)
        self.assertAlmostEqual(result['mINP'], 0.5)
        self.assertEqual(result['Rank-1'], 0)
        self.assertEqual(result['Rank-5'], 1)

    def test_perfect_ranking(self):
        labels = np.stack([self.exact, self.exact, self.near, self.far])
        result = query_metrics(self.query, labels, np.arange(4))
        for value in result.values():
            self.assertEqual(value, 1)

    def test_all_exact_degenerate_normalization(self):
        result = query_metrics(self.query, np.stack([self.exact] * 3), np.zeros(3))
        self.assertEqual(result['mADM'], 1)

    def test_stable_ties(self):
        labels = np.stack([self.far, self.exact])
        result = query_metrics(self.query, labels, [0, 0])
        self.assertEqual(result['mAP'], 0.5)
        self.assertEqual(result['Rank-1'], 0)

    def test_missing_exact_match_rejected(self):
        with self.assertRaises(ValueError):
            query_metrics(self.query, np.stack([self.near]), [0])

    def test_matrix_matches_direct_l1(self):
        rng = np.random.default_rng(4)
        queries = rng.integers(0, 2, (5, 40))
        probs = rng.random((7, 40), dtype=np.float32)
        expected = np.abs(queries[:, None, :] - probs[None, :, :]).sum(2)
        np.testing.assert_allclose(distance_matrix(queries, probs, 2), expected, atol=1e-5)

    def test_evaluator_matches_scalar_definition(self):
        labels = np.stack([self.near, self.exact, self.far, self.exact])
        queries = np.stack([self.query, self.far])
        probabilities = np.clip(labels * 0.6 + 0.15, 0, 1)
        rows = [query_metrics(q, labels, d) for q, d in zip(queries, l1_distances(queries, probabilities))]
        result = evaluate_retrieval(queries, labels, probabilities, 1)
        for key in result:
            self.assertAlmostEqual(result[key], np.mean([row[key] for row in rows]))

    def test_mixture_distances_match_rescoring(self):
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
        from rescore_lodo import calibrate
        from upar.retrieval import bce_distances, mixture_distance_matrix
        rng = np.random.default_rng(1)
        queries = (rng.random((5, 40)) < 0.3).astype(np.float32)
        members = rng.random((3, 7, 40))
        members[0] = calibrate(members[0], np.zeros(40), 0.05, 0.0)  # one corrected member
        mean = members.mean(0)
        each = np.stack([bce_distances(queries, m.astype(np.float32)) for m in members]).astype(np.float64)
        low = each.min(0)
        mixed = low - 4 * np.log(np.exp(-(each - low) / 4).mean(0))
        expected = 0.5 * bce_distances(queries, mean.astype(np.float32)) + 0.5 * mixed
        result = mixture_distance_matrix(queries, mean, members, chunk_size=2, temperature=4, weight=0.5)
        self.assertTrue(np.allclose(result, expected, rtol=1e-4, atol=1e-3))
        plain = mixture_distance_matrix(queries, mean, members, temperature=4, weight=0.0)
        self.assertTrue(np.allclose(plain, bce_distances(queries, mean.astype(np.float32)), rtol=1e-4, atol=1e-3))

    def test_invalid_probability_rejected(self):
        with self.assertRaises(ValueError):
            l1_distances([self.query], np.full((1, 40), np.nan))


if __name__ == '__main__':
    unittest.main()
