import unittest
import pandas as pd

from guacml.util.time_series_util import analyze_frequency


class TestTimeSeriesUtil(unittest.TestCase):
    def test_analyze_frequency_happy(self):
        df = pd.DataFrame({
            'date':
                list(pd.date_range(pd.datetime(2015, 6, 15), pd.datetime(2015, 6, 18))) +
                list(pd.date_range(pd.datetime(2015, 6, 1, 1), pd.datetime(2015, 6, 4, 1))),
            'series_key': ['a'] * 4 + ['b'] * 4
        })
        ts_config = {'date_split_col': 'date', 'series_key_cols': ['series_key']}

        frequency = analyze_frequency(df, ts_config)
        self.assertEqual(frequency, pd.Timedelta(days=1))

    def test_analyse_frquency_exception(self):
        df = pd.DataFrame({
            'date':
                list(pd.date_range(pd.datetime(2015, 6, 15), pd.datetime(2015, 6, 18))) +
                list(pd.date_range(pd.datetime(2015, 6, 1, 1), pd.datetime(2015, 6, 4, 1))),
            'series_key': ['a'] * 8
        })
        ts_config = {'date_split_col': 'date', 'series_key_cols': ['series_key']}

        with self.assertRaises(ValueError):
            analyze_frequency(df, ts_config)
