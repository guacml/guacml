import guacml as guac
import os
import pandas as pd
import unittest
from sklearn.ensemble import RandomForestClassifier

class TestDataset(unittest.TestCase):
    def load_dataset(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        return guac.Dataset(dir_path + '/fixtures/titanic.csv', 'Survived')

    def test_dataset(self):
        ds = self.load_dataset()

        self.assertIsInstance(ds.df, pd.DataFrame)

    def test_run(self):
        ds = self.load_dataset()

        ds.run(1)
        result = ds.model_results

        self.assertEquals(3, len(result))
        self.assertTrue(0 < result['random_forest'].training_error < 1)
        self.assertTrue(0 < result['random_forest'].cv_error < 1)

    def test_accuracy(self):
        ds = self.load_dataset()

        ds.run(1)
        result = ds.model_results
        self.assertAlmostEqual(0.8268, result['random_forest'].holdout_accuracy, delta=0.05)
