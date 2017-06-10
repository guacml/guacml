from guacml import GuacMl
import os
import pandas as pd
import unittest


class TestDataset(unittest.TestCase):
    def load_dataset(self, fixture='titanic', target='Survived', eval_metric=None):
        dir_path = os.path.dirname(os.path.realpath(__file__))

        df = pd.read_csv('{0}/fixtures/{1}.csv'.format(dir_path, fixture))
        guac = GuacMl(df, target, eval_metric=eval_metric)
        guac.clear_previous_runs()
        return guac

    def test_dataset(self):
        ds = self.load_dataset()
        self.assertIsInstance(ds.data.df, pd.DataFrame)

    def test_run(self):
        guac = self.load_dataset()

        guac.run(1)
        result = guac.model_results

        self.assertEqual(3, len(result))
        self.assertAlmostEqual(100, result['random_forest'].training_error, delta=150)
        self.assertAlmostEqual(100, result['random_forest'].cv_error, delta=150)

    def test_accuracy(self):
        guac = self.load_dataset(eval_metric='accuracy')
        guac.run(1)
        result = guac.model_results
        self.assertAlmostEqual(-0.8, result['random_forest'].holdout_error, delta=0.2)

    def test_boolean_column(self):
        guac = self.load_dataset('boolean')
        guac.run(1)
        result = guac.model_results
        self.assertAlmostEqual(0.5, result['random_forest'].holdout_error, delta=0.2)

    def test_date_splitter(self):
        guac = self.load_dataset(fixture='bike_sharing', target='count')
        guac.run(1, 'datetime')
        result = guac.model_results
        self.assertEqual(3, len(result))


