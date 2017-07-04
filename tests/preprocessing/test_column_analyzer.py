import unittest
import numpy as np
import pandas as pd

from guacml.enums import ColType
from guacml.preprocessing.column_analyzer import ColumnAnalyzer


class TestColumnAnalyzer(unittest.TestCase):
    def test_float_column_only_bool(self):
        df = pd.DataFrame({'a': [0, 1, None]})
        df.a = df.a.astype(float)
        self.assertEqual(df['a'].values.dtype, float)

        analyzer = ColumnAnalyzer()
        meta = analyzer.analyze(df)
        self.assertEqual(meta.loc['a'].type, ColType.BINARY)

    def test_float_column_only_integers(self):
        df = pd.DataFrame({'a': [1, 100, 100000, None]})
        df.a = df.a.astype(float)
        self.assertEqual(df['a'].values.dtype, float)

        analyzer = ColumnAnalyzer()
        meta = analyzer.analyze(df)
        self.assertEqual(meta.loc['a'].type, ColType.ORDINAL)

    def test_mixed_int_str_col(self):
        df = pd.DataFrame({'a': [0, '0', 'a']})
        analyzer = ColumnAnalyzer()
        meta = analyzer.analyze(df)
        self.assertEqual(meta.loc['a'].type, ColType.CATEGORICAL)
        self.assertEqual(df.drop_duplicates().shape[0], 2)

    def test_int8(self):
        df = pd.DataFrame({'a': [0, 5, 2]}, dtype=np.int8)
        analyzer = ColumnAnalyzer()
        meta = analyzer.analyze(df)
        self.assertEqual(meta.loc['a'].type, ColType.ORDINAL)

    def test_float16(self):
        df = pd.DataFrame({'a': [0.1, 2.3]}, dtype=np.float16)
        analyzer = ColumnAnalyzer()
        meta = analyzer.analyze(df)
        self.assertEqual(meta.loc['a'].type, ColType.NUMERIC)
