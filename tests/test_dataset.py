from guacml import GuacMl
import os
import pandas as pd
import unittest


class TestDataset(unittest.TestCase):
    def load_dataset(self, fixture='titanic', target='Survived', eval_metric=None):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        df = pd.read_csv(dir_path + '/fixtures/' + fixture + '.csv')
        return GuacMl(df, 'Survived', eval_metric=eval_metric)


    def test_dataset(self):
        ds = self.load_dataset()
        self.assertIsInstance(ds.data.df, pd.DataFrame)

    def test_run(self):
        ds = self.load_dataset()

        ds.run(1)
        result = ds.model_results

        self.assertEqual(3, len(result))
        self.assertAlmostEqual(100, result['random_forest'].training_error, delta=150)
        self.assertAlmostEqual(100, result['random_forest'].cv_error, delta=150)
        ds.clear_prev_runs()

    def test_accuracy(self):
        ds = self.load_dataset(eval_metric='accuracy')
        ds.run(1)
        result = ds.model_results
        self.assertAlmostEqual(-0.8, result['random_forest'].holdout_error, delta=0.2)
        ds.clear_prev_runs()

    def test_boolean_column(self):
        ds = self.load_dataset('boolean')

        ds.run(1)
        result = ds.model_results
        self.assertAlmostEqual(0.5, result['random_forest'].holdout_error, delta=0.2)
        ds.clear_prev_runs()

    def test_date_splitter(self):
        ds = self.load_dataset(fixture='bike_sharing', target='count')
        ds.run(1, 'datetime')
        result = ds.model_results
        self.assertEqual(3, len(result))
        ds.clear_prev_runs()

