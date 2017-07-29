import unittest
import numpy as np
import pandas as pd
import logging

from tests.test_util import load_config
from guacml.preprocessing.column_analyzer import ColumnAnalyzer


class TestColumnAnalyzer(unittest.TestCase):
    def build_analyzer(self, more_config={}):
        config = load_config()
        config.update(more_config)
        logger = logging.getLogger(__name__)

        return ColumnAnalyzer(config, logger)

    def test_float_column_only_bool(self):
        df = pd.DataFrame({'a': [0, 1, None]})
        df.a = df.a.astype(float)
        self.assertEqual(df['a'].values.dtype, float)

        analyzer = self.build_analyzer()
        meta = analyzer.analyze(df)
        self.assertEqual(meta.loc['a'].type, 'binary')

    def test_float_column_only_integers(self):
        df = pd.DataFrame({'a': [1, 100, 100000, None]})
        df.a = df.a.astype(float)
        self.assertEqual(df['a'].values.dtype, float)

        analyzer = self.build_analyzer()
        meta = analyzer.analyze(df)
        self.assertEqual(meta.loc['a'].type, 'ordinal')

    def test_mixed_int_str_col(self):
        df = pd.DataFrame({'a': [0, '0', 'a']})
        analyzer = self.build_analyzer()
        meta = analyzer.analyze(df)
        self.assertEqual(meta.loc['a'].type, 'categorical')
        self.assertEqual(df.drop_duplicates().shape[0], 2)

    def test_int8(self):
        df = pd.DataFrame({'a': [0, 5, 2]}, dtype=np.int8)
        analyzer = self.build_analyzer()
        meta = analyzer.analyze(df)
        self.assertEqual(meta.loc['a'].type, 'ordinal')

    def test_float16(self):
        df = pd.DataFrame({'a': [0.1, 2.3]}, dtype=np.float16)
        analyzer = self.build_analyzer()
        meta = analyzer.analyze(df)
        self.assertEqual(meta.loc['a'].type, 'numeric')

    def test_override_column_type(self):
        df = pd.DataFrame({'a': ['b', 'c']})
        analyzer = self.build_analyzer({'column_types': {'a': 'text'}})
        meta = analyzer.analyze(df)
        self.assertEqual(meta.loc['a'].type, 'text')
