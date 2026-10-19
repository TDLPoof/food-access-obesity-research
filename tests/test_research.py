"""Check data contracts and evidence that evaluation is geographically held out."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from model import FEATURE_SETS, load_data, grouped_splits, metrics


class ResearchChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_data(ROOT / 'data/processed/county_obesity_2019_model_data.csv')

    def test_dataset_contract(self):
        self.assertEqual(len(self.data), 3110)
        self.assertTrue(self.data.fips.str.len().eq(5).all())
        self.assertTrue(self.data.fips.is_unique)

    def test_rejects_duplicate_counties_and_missing_measurements(self):
        for bad in [pd.concat([self.data, self.data.iloc[:1]]), self.data.assign(low_access_share=np.nan)]:
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / 'bad.csv'
                bad.to_csv(path, index=False)
                with self.assertRaises(ValueError):
                    load_data(path)

    def test_nested_geographic_separation(self):
        seen = []
        for train, test in grouped_splits(self.data.state, 5):
            outer_train = self.data.iloc[train]
            outer_test_states = set(self.data.iloc[test].state)
            self.assertFalse(set(outer_train.state) & outer_test_states)
            for a, b in grouped_splits(outer_train.state, 3):
                self.assertFalse(set(outer_train.iloc[a].state) & set(outer_train.iloc[b].state))
                self.assertFalse(set(outer_train.iloc[b].state) & outer_test_states)
            seen.extend(test)
        self.assertEqual(sorted(seen), list(range(len(self.data))))

    def test_predictors_exclude_outcome_and_redundant_index(self):
        for features in FEATURE_SETS.values():
            self.assertFalse({'obesity_adj_pct', 'y_prop', 'fips', 'state'} & set(features))
            if 'swamp_logratio' in features:
                self.assertNotIn('healthy_density', features)
                self.assertNotIn('unhealthy_density', features)

    def test_published_predictions_and_metrics_agree(self):
        output = ROOT / 'results/portfolio'
        predictions = pd.read_csv(output / 'predictions.csv', dtype={'fips': str})
        summary = pd.read_csv(output / 'summary.csv')
        self.assertEqual(len(summary), 9)
        for row in summary.itertuples():
            part = predictions[(predictions.model == row.model) & (predictions.feature_set == row.feature_set)]
            self.assertEqual(len(part), len(self.data))
            self.assertEqual(set(part.fips), set(self.data.fips))
            self.assertTrue(part.fips.is_unique)
            for metric, actual in metrics(part.observed, part.predicted).items():
                self.assertAlmostEqual(getattr(row, metric), actual, places=10)

    def test_published_fold_audit(self):
        audit = json.loads((ROOT / 'results/portfolio/fold_audit.json').read_text())
        for fold in audit:
            outer_train, outer_test = set(fold['train_states']), set(fold['test_states'])
            self.assertFalse(outer_train & outer_test)
            for inner in fold['inner']:
                a, b = set(inner['train_states']), set(inner['validation_states'])
                self.assertFalse(a & b)
                self.assertEqual(a | b, outer_train)
                self.assertFalse((a | b) & outer_test)


if __name__ == '__main__':
    unittest.main()
