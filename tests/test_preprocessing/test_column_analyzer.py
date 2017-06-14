import unittest
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

