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

        result = ds.run()

        self.assertEquals(3, len(result))
        self.assertIsInstance(result['random_forest'], RandomForestClassifier)
