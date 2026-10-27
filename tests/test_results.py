import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from upar.evaluation import evaluate_split
from upar.results import save_predictions, load_predictions, append_results


class ResultTests(unittest.TestCase):
    def test_cache_roundtrip_preserves_queries_domains_and_metrics(self):
        labels = np.zeros((3, 40), dtype=np.uint8)
        labels[1, 0] = 1
        queries = labels[:2].copy()
        ids = np.array([0, 1, 0])
        names = ['A/0.jpg', 'B/1.jpg', 'A/2.jpg']
        probs = labels * 0.8 + 0.1
        expected = evaluate_split(queries, labels, ids, probs, names)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'val_predictions.npz'
            save_predictions(path, probs, labels, queries, ids, names, {'epoch': 3})
            data = load_predictions(path)
            self.assertEqual(data['domain'].tolist(), ['A', 'B', 'A'])
            self.assertEqual(data['metadata']['epoch'], 3)
            actual = evaluate_split(data['queries'], data['labels'], data['ids'], data['probs'], data['image_names'])
            self.assertEqual(actual, expected)
            results = Path(folder) / 'results.csv'
            append_results(results, actual, {'epoch': 3, 'retrieval': 'l1'})
            append_results(results, actual, {'epoch': 4, 'retrieval': 'bce'})
            with results.open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 6)
            self.assertEqual(rows[0]['gallery'], 'all')
            self.assertEqual(rows[-1]['epoch'], '4')
