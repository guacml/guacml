import guacml as guac
import os
import pandas as pd
import unittest

class TestDataset(unittest.TestCase):
    def test_dataset(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        ds = guac.Dataset(dir_path + '/fixtures/titanic.csv')
        self.assertIsInstance(ds.df, pd.DataFrame)
