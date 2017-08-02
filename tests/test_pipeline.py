import unittest
import base64

from guacml.pipeline import Pipeline
from tests.test_util import load_dataset, read_fixture


class TestPipeline(unittest.TestCase):

    def quick_run(self):
        guac = load_dataset()

        for s in ['fill_na', 'random_forest', 'one_hot_encode', 'linear_model']:
            guac.tree.delete_step(s)

        guac.run(1)

        return guac

    def test_predict(self):
        guac = self.quick_run()
        pipeline = Pipeline(guac, 'xgboost')
        test_set = read_fixture().head()
        actual = pipeline.predict(test_set).tolist()

        self.assertEqual(5, len(actual))

        for a in actual:
            self.assertTrue(0 < a < 1)

    def test_serialize(self):
        guac = self.quick_run()
        pipeline = Pipeline(guac, 'xgboost')
        state = pipeline.serialize()

        self.assertEqual('xgboost', state['name'])

        self.assertEqual(pipeline.config, state['config'])
        self.assertIsNot(pipeline.config, state['config'])

        self.assertEqual(pipeline.features, state['features'])
        self.assertIsNot(pipeline.features, state['features'])

        self.assertEqual('guacml.models.xgboost.XGBoost', state['model']['class'])
        self.assertIsInstance(state['model']['state'], str)
        self.assertIsInstance(base64.b64decode(state['model']['state'].encode()), bytes)

        self.assertEqual(pipeline.tree.children, state['tree']['children'])
        self.assertIsNot(pipeline.tree.children, state['tree']['children'])

        step = state['tree']['steps']['encode_labels']
        self.assertEqual(list(pipeline.tree.steps), list(state['tree']['steps']))
        self.assertEqual('guacml.preprocessing.feature_generation.label_encoder.LabelEncoder', step['class'])
        self.assertEqual(['female', 'male'], step['state']['classes']['Sex'])
        self.assertIsNot(pipeline.tree.get_step('encode_labels').state, step['state'])
