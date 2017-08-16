from guacml.dataset import Dataset
from guacml.preprocessing.feature_generation.label_encoder import LabelEncoder
import pandas as pd
import numpy as np
import unittest
import logging
import tests.test_util as test_util


class TestLabelEncoder(unittest.TestCase):

    def _get_logger(self):
        return logging.getLogger(__name__)

    def test_access_classes_per_column(self):
        config = test_util.load_config()
        encoder = LabelEncoder(config, self._get_logger())
        input = pd.DataFrame({'sex': ['male', 'female', None],
                             'nationality': ['Polish', 'German', 'Polish'], 'age': [21, 23, 25]})
        metadata = pd.DataFrame({
            'type': ['categorical', 'categorical', 'numeric'],
            'derived_from': [None, None, None],
            'n_unique': [3, 3, 3],
            'n_na': [1, 0, 0],
            'n_blank': [1, 0, 0]
        }, index=['sex', 'nationality', 'age'])
        dataset = Dataset(input, metadata)

        encoder.execute_inplace(dataset)

        self.assertEqual(2, len(encoder.state['classes']))
        self.assertEqual(['female', 'male'], list(encoder.state['classes']['sex']))
        self.assertEqual(['German', 'Polish'], list(encoder.state['classes']['nationality']))

    def test_return_nans_for_unknown_labels(self):
        config = test_util.load_config()
        encoder = LabelEncoder(config, self._get_logger())
        train_input = pd.DataFrame({'sex': ['male', 'female', None],
                                    'nationality': ['Polish', 'German', 'Polish'], 'age': [21, 23, 25]})
        metadata = pd.DataFrame({
            'type': ['categorical', 'categorical', 'numeric'],
            'derived_from': [None, None, None],
            'n_unique': [3, 3, 3],
            'n_na': [1, 0, 0],
            'n_blank': [1, 0, 0]
        }, index=['sex', 'nationality', 'age'])
        train_set = Dataset(train_input, metadata)

        encoder.execute_inplace(train_set)

        test_input = pd.DataFrame({'sex': ['male', 'female', None],
                                   'nationality': ['Polish', 'German', 'French'], 'age': [21, 23, 25]})
        test_set = Dataset(test_input, metadata)

        encoder.execute_inplace(test_set)

        actual = test_set.df.nationality.tolist()
        self.assertEqual([1.0, 0.0], actual[0:2])
        self.assertTrue(np.isnan(actual[2]))
