from guacml.dataset import Dataset
from guacml.preprocessing.column_analyzer import ColType
from guacml.preprocessing.feature_generation.one_hot_encoder import OneHotEncoder
import pandas as pd
import unittest
import tests.test_util as test_util


class TestOneHotEncoder(unittest.TestCase):

    def test_encoding(self):
        config = test_util.load_config()
        enc = OneHotEncoder(config['pre_processing'])
        input = pd.DataFrame({'a': [3, 1, 2]})
        metadata = pd.DataFrame({
            'col_name': ['a'],
            'type': [ColType.INT_ENCODING],
            'derived_from': [None],
            'n_unique': [input['a'].nunique()],
            'n_na': [0],
            'n_blank': [0]
        })
        dataset = Dataset(input, metadata)
        data = enc.execute(dataset)
        output = data.df
        meta_out = data.metadata

        self.assertEqual(output.shape, (3, 4))
        self.assertEqual(output.columns[1], 'a_one_hot_1')
        self.assertEqual(output['a_one_hot_1'].iloc[0], 0)
        self.assertEqual(output['a_one_hot_3'].iloc[0], 1)

        self.assertEqual(meta_out.shape[0], 4)
        self.assertEqual(meta_out['col_name'].iloc[1], 'a_one_hot_1')
        self.assertEqual(meta_out['type'].iloc[1], ColType.BINARY)
