from guacml.dataset import Dataset
from guacml.preprocessing.feature_generation.one_hot_encoder import OneHotEncoder
import pandas as pd
import unittest
import logging
import tests.test_util as test_util


class TestOneHotEncoder(unittest.TestCase):

    def _get_logger(self):
        return logging.getLogger(__name__)

    def test_encoding(self):
        config = test_util.load_config()
        enc = OneHotEncoder(config, self._get_logger())
        input = pd.DataFrame({'a': [3, 1, 2], 'b': [1.1, 1.2, 1.3]},
                             index=['row_1', 'row_2', 'row_3'])
        metadata = pd.DataFrame({
            'type': ['int_encoding', 'numeric'],
            'derived_from': [None, None],
            'n_unique': [input['a'].nunique(), input['b'].nunique()],
            'n_na': [0, 0],
            'n_blank': [0, 0]
        }, index=['a', 'b'])
        dataset = Dataset(input, metadata)
        data = enc.execute(dataset)
        output = data.df
        meta_out = data.metadata

        self.assertEqual(output.shape, (3, 5))
        self.assertEqual(output.columns[2], 'a_one_hot_1')
        self.assertEqual(output['a_one_hot_1'].iloc[0], 0)
        self.assertEqual(output['a_one_hot_3'].iloc[0], 1)

        self.assertEqual(meta_out.shape[0], 5)
        self.assertEqual(meta_out.index[2], 'a_one_hot_1')
        self.assertEqual(meta_out['type'].iloc[2], 'binary')

    def test_single_zero_column(self):
        """ This was causing an exception in the One-Hot-Encoder."""
        config = test_util.load_config()
        enc = OneHotEncoder(config, self._get_logger())
        input = pd.DataFrame({'a': [0.0, 0.0, 0.0]})
        metadata = pd.DataFrame({
            'col_name': ['a'],
            'type': ['int_encoding'],
            'derived_from': [None],
            'n_unique': [input['a'].nunique()],
            'n_na': [0],
            'n_blank': [0]
        })
        dataset = Dataset(input, metadata)
        data = enc.execute(dataset)

        output = data.df
        meta_out = data.metadata

        self.assertEqual(output.shape, (3, 1))
        self.assertEqual(meta_out.shape[0], 1)
