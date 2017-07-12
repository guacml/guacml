from guacml.preprocessing.feature_generation.historical_medians import HistoricalMedians
import numpy as np
import unittest
import tests.test_util as test_util


class TestHistoricalMedians(unittest.TestCase):

    def test_medians(self):
        guac = test_util.load_dataset('timeseries', target='Sales')
        guac.make_time_series('Date', series_key_cols='Store', prediction_length=2)

        medians = HistoricalMedians(guac.config)
        out = medians.execute(guac.data)

        self.assertTrue(np.isnan(out.df['Sales_median_2'].iloc[0]))
        self.assertTrue(np.isnan(out.df['Sales_median_2'].iloc[1]))
        self.assertTrue(np.isnan(out.df['Sales_median_2'].iloc[2]))
        self.assertAlmostEqual(out.df['Sales_median_2'].iloc[3], 43552444.5, delta=1)

    def test_medians_no_group_keys(self):
        guac = test_util.load_dataset('bike_sharing', target='count')
        guac.make_time_series('datetime', prediction_length=1)

        medians = HistoricalMedians(guac.config)
        out = medians.execute(guac.data)

        self.assertTrue(np.isnan(out.df['count_median_1'].iloc[0]))
        self.assertAlmostEqual(out.df['count_median_1'].iloc[1], 16, delta=1)

    # ToDo: test the metadata
