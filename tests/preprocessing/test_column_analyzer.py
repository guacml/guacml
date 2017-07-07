import unittest
import numpy as np
import pandas as pd
import logging

from guacml.enums import ColType
from guacml.preprocessing.column_analyzer import ColumnAnalyzer


class TestColumnAnalyzer(unittest.TestCase):
    def build_analyzer(self):
        logger = logging.getLogger(__name__)
        return ColumnAnalyzer(logger)

    def test_float_column_only_bool(self):
        df = pd.DataFrame({'a': [0, 1, None]})
        df.a = df.a.astype(float)
        self.assertEqual(df['a'].values.dtype, float)

        analyzer = self.build_analyzer()
        meta = analyzer.analyze(df)
        self.assertEqual(meta.loc['a'].type, ColType.BINARY)

    def test_float_column_only_integers(self):
        df = pd.DataFrame({'a': [1, 100, 100000, None]})
        df.a = df.a.astype(float)
        self.assertEqual(df['a'].values.dtype, float)

        analyzer = self.build_analyzer()
        meta = analyzer.analyze(df)
        self.assertEqual(meta.loc['a'].type, ColType.ORDINAL)

    def test_mixed_int_str_col(self):
        df = pd.DataFrame({'a': [0, '0', 'a']})
        analyzer = self.build_analyzer()
        meta = analyzer.analyze(df)
        self.assertEqual(meta.loc['a'].type, ColType.CATEGORICAL)
        self.assertEqual(df.drop_duplicates().shape[0], 2)

    def test_int8(self):
        df = pd.DataFrame({'a': [0, 5, 2]}, dtype=np.int8)
        analyzer = self.build_analyzer()
        meta = analyzer.analyze(df)
        self.assertEqual(meta.loc['a'].type, ColType.ORDINAL)

    def test_float16(self):
        df = pd.DataFrame({'a': [0.1, 2.3]}, dtype=np.float16)
        analyzer = self.build_analyzer()
        meta = analyzer.analyze(df)
        self.assertEqual(meta.loc['a'].type, ColType.NUMERIC)
