from guacml import GuacMl
import pandas as pd
import unittest

from tests.test_util import load_dataset


class TestDataset(unittest.TestCase):
    def test_dataset(self):
        ds = load_dataset()
        self.assertIsInstance(ds.data.df, pd.DataFrame)

    def test_run(self):
        guac = load_dataset()

        guac.run(1)
        result = guac.model_results

        self.assertEqual(3, len(result))
        self.assertAlmostEqual(100, result['random_forest'].training_error, delta=150)
        self.assertAlmostEqual(100, result['random_forest'].cv_error, delta=150)

    def test_accuracy(self):
        guac = load_dataset(eval_metric='accuracy')
        guac.run(1)
        result = guac.model_results
        self.assertAlmostEqual(-0.8, result['random_forest'].holdout_error, delta=0.2)

    def test_boolean_column(self):
        guac = load_dataset('boolean')
        guac.run(1)
        result = guac.model_results
        self.assertAlmostEqual(0.5, result['random_forest'].holdout_error, delta=0.2)

    def test_date_splitter(self):
        guac = load_dataset(fixture='bike_sharing', target='count')
        guac.make_time_series('datetime')
        guac.run(1)
        result = guac.model_results
        self.assertEqual(3, len(result))

    def test_timeseries(self):
        guac = load_dataset(fixture='timeseries', target='Sales')
        guac.make_time_series('Date', prediction_length=2)
        guac.run(1)
        result = guac.model_results
        self.assertEqual(3, len(result))

    def test_dataset_with_non_canonical_index(self):
        df = pd.DataFrame({'a': range(100), 'b': [x + 0.5 for x in range(100)]},
                          index=range(100, 200))
        guac = GuacMl(df, 'b')
        guac.clear_previous_runs()
        guac.run(1)
        self.assertAlmostEqual(guac.model_results['linear_model'].holdout_error, 0.0, delta=1e-5)

    def test_disabled_feature_reduction(self):
        guac = load_dataset()
        without_feature_reduction = load_dataset(config={'model_manager': {'reduce_features': False}})

        guac.run(1)
        without_feature_reduction.run(1)
        guac_result = guac.model_results
        wofr_result = without_feature_reduction.model_results

        self.assertLess(len(guac_result['linear_model'].features), len(wofr_result['linear_model'].features))
