import unittest
import base64
import json

from guacml.pipeline import Pipeline
from guacml.models.xgboost import XGBoost
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
        pipeline = guac.get_pipeline('xgboost')
        test_set = read_fixture().head()
        actual = pipeline.predict(test_set).tolist()

        self.assertEqual(5, len(actual))

        for a in actual:
            self.assertTrue(0 < a < 1)

    def test_serialize(self):
        guac = self.quick_run()
        pipeline = guac.get_pipeline('xgboost')
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

    def test_deserialize(self):
        guac = self.quick_run()
        pipeline = guac.get_pipeline('xgboost')
        cloned_state = json.loads(json.dumps(pipeline.serialize()))
        cloned_pipeline = Pipeline.deserialize(cloned_state, guac.logger)
        test_set = read_fixture().head()

        self.assertEqual(pipeline.name, cloned_pipeline.name)
        self.assertEqual(pipeline.config, cloned_pipeline.config)
        self.assertEqual(pipeline.features, cloned_pipeline.features)
        self.assertIsInstance(cloned_pipeline.model, XGBoost)
        self.assertEqual(pipeline.tree.children, cloned_pipeline.tree.children)
        self.assertEqual(pipeline.tree.get_step('encode_labels').state,
                         cloned_pipeline.tree.get_step('encode_labels').state)
        self.assertEqual(pipeline.predict(test_set).tolist(),
                         cloned_pipeline.predict(test_set).tolist())
