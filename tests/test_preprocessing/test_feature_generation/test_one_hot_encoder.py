from guacml.preprocessing.column_analyzer import ColType
from guacml.preprocessing.feature_generation.one_hot_encoder import OneHotEncoder
import pandas as pd
import unittest


class TestOneHotEncoder(unittest.TestCase):

    def test_encoding(self):
        enc = OneHotEncoder()
        input = pd.DataFrame({'a': [3, 1, 2]})
        metadata = pd.DataFrame({
            'col_name': ['a'],
            'type': [ColType.INT_ENCODING],
            'derived_from': [None],
            'n_unique': [input['a'].nunique()],
            'n_na': [0],
            'n_blank': [0]
        })
        output, meta_out = enc.execute(input, metadata)

        self.assertEquals(output.shape, (3, 4))
        self.assertEquals(output.columns[1], 'a_one_hot_1')
        self.assertEquals(output['a_one_hot_1'].iloc[0], 0)
        self.assertEquals(output['a_one_hot_3'].iloc[0], 1)

        self.assertEquals(meta_out.shape[0], 4)
        self.assertEquals(meta_out['col_name'].iloc[1], 'a_one_hot_1')
        self.assertEquals(meta_out['type'].iloc[1], ColType.BINARY)
