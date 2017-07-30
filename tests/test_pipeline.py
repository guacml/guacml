import unittest

from guacml.pipeline import Pipeline
from tests.test_util import load_dataset, read_fixture


class TestPipeline(unittest.TestCase):

    def test_predict(self):
        guac = load_dataset()

        for s in ['fill_na', 'random_forest', 'one_hot_encode', 'linear_model']:
            guac.tree.delete_step(s)

        guac.run(1)

        pipeline = Pipeline(guac, 'xgboost')
        test_set = read_fixture().head()
        actual = pipeline.predict(test_set).tolist()

        self.assertEqual(5, len(actual))

        for a in actual:
            self.assertTrue(0 < a < 1)
