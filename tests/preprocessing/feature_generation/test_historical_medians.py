from guacml.preprocessing.feature_generation.historical_medians import HistoricalMedians
import numpy as np
import pandas as pd
import unittest
import tests.test_util as test_util
from guacml import GuacMl


class TestHistoricalMedians(unittest.TestCase):

    def test_medians(self):
        guac = test_util.load_dataset('timeseries', target='Sales')
        guac.make_time_series('Date', series_key_cols='Store', prediction_length=2)

        medians = HistoricalMedians([1], guac.config, guac.logger)
        out = medians.execute(guac.data)

        out.df = out.df.sort_values(['Store', 'Date'])
        self.assertTrue(np.isnan(out.df['Sales_median_2'].iloc[0]))
        self.assertTrue(np.isnan(out.df['Sales_median_2'].iloc[1]))
        self.assertAlmostEqual(out.df['Sales_median_2'].iloc[2], 0, delta=0.01)
        self.assertAlmostEqual(out.df['Sales_median_2'].iloc[3], 43552444.5, delta=1)

    def test_medians_no_series_keys(self):
        guac = test_util.load_dataset('bike_sharing', target='count')
        guac.make_time_series('datetime', prediction_length=1, frequency=pd.DateOffset(hours=1))

        medians = HistoricalMedians([1], guac.config, guac.logger)
        out = medians.execute(guac.data)

        out.df = out.df.sort_values('datetime')
        self.assertTrue(np.isnan(out.df['count_median_1'].iloc[0]))
        self.assertAlmostEqual(out.df['count_median_1'].iloc[1], 16, delta=1)

    def test_medians_for_gaps(self):
        df = pd.DataFrame({
            'date': list(pd.date_range(pd.datetime(2015, 6, 15),
                                       pd.datetime(2015, 6, 20))),
            'value': range(6)
        })
        df = df.iloc[[0, 2, 3, 4, 5]]
        guac = GuacMl(df, 'value')
        guac.make_time_series('date', prediction_length=1)
        medians = HistoricalMedians([3], guac.config, guac.logger)
        out = medians.execute(guac.data)
        self.assertTrue(np.isnan(out.df['value_median_3'].iloc[0]))
        self.assertEqual(out.df['value_median_3'].iloc[1], 0)
        self.assertEqual(out.df['value_median_3'].iloc[2], 1)
        self.assertEqual(out.df['value_median_3'].iloc[3], 2.5)
        self.assertEqual(out.df['value_median_3'].iloc[4], 3)

    def test_medians_series_and_group_keys_simple(self):
        df = pd.DataFrame({
            'date':
                list(pd.date_range(pd.datetime(2015, 6, 15), pd.datetime(2015, 6, 20))) +
                list(pd.date_range(pd.datetime(2015, 6, 15), pd.datetime(2015, 6, 20))),
            'series_key': ['a'] * 6 + ['b'] * 6,
            'group_key': ['uneven', 'even'] * 6,
            'value': range(12)
        })
        guac = GuacMl(df, 'value')
        guac.make_time_series('date', prediction_length=1, series_key_cols='series_key')
        medians = HistoricalMedians([2], guac.config, guac.logger, group_keys='group_key')
        out = medians.execute(guac.data)

        out.df = out.df.sort_values(['series_key', 'group_key', 'date'])
        self.assertTrue(np.isnan(out.df['value_median_2_by_group_key'].iloc[0]))
        self.assertEqual(out.df['value_median_2_by_group_key'].iloc[1], 1)
        self.assertEqual(out.df['value_median_2_by_group_key'].iloc[2], 2)
        self.assertTrue(np.isnan(out.df['value_median_2_by_group_key'].iloc[3]))
        self.assertEqual(out.df['value_median_2_by_group_key'].iloc[4], 0)
        self.assertEqual(out.df['value_median_2_by_group_key'].iloc[5], 1)
        self.assertTrue(np.isnan(out.df['value_median_2_by_group_key'].iloc[6]))

    def test_medians_series_and_group_keys(self):
        guac = test_util.load_dataset('timeseries', target='Sales')
        guac.make_time_series('Date', series_key_cols='Store', prediction_length=2)

        guac.data.df['weekday'] = guac.data.df['Date'].dt.weekday
        medians = HistoricalMedians([1], guac.config, guac.logger, group_keys='weekday')
        out = medians.execute(guac.data)

        out.df = out.df.sort_values(['Store', 'weekday', 'Date'])
        self.assertTrue(np.isnan(out.df['Sales_median_2_by_weekday'].iloc[0]))
        self.assertTrue(np.isnan(out.df['Sales_median_2_by_weekday'].iloc[1]))
        self.assertAlmostEqual(out.df['Sales_median_2_by_weekday'].iloc[2], 106688241, delta=1)
        self.assertAlmostEqual(out.df['Sales_median_2_by_weekday'].iloc[3], 92099328.5, delta=1)

    # ToDo: test the metadata
