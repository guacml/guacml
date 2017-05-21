import guacml as guac
import os
import pandas as pd
import unittest


class TestDataset(unittest.TestCase):
    def load_dataset(self, eval_metric=None):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        return guac.Dataset(dir_path + '/fixtures/titanic.csv', 'Survived', eval_metric=eval_metric)

    def test_dataset(self):
        ds = self.load_dataset()

        self.assertIsInstance(ds.df, pd.DataFrame)

    def test_run(self):
        ds = self.load_dataset()

        ds.run(1)
        result = ds.model_results

        self.assertEquals(3, len(result))
        self.assertAlmostEqual(100, result['random_forest'].training_error, delta=150)
        self.assertAlmostEqual(100, result['random_forest'].cv_error, delta=150)

    def test_accuracy(self):
        ds = self.load_dataset(eval_metric='accuracy')

        ds.run(1)
        result = ds.model_results
        self.assertAlmostEqual(-0.8, result['random_forest'].holdout_error, delta=0.2)
