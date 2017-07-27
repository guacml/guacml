from guacml.preprocessing.feature_generation.lagged_target import LaggedTarget
import numpy as np
import unittest
import tests.test_util as test_util


class TestLaggedTarget(unittest.TestCase):

    def test_lagged_target(self):
        guac = test_util.load_dataset('timeseries', target='Sales')
        guac.make_time_series('Date', series_key_cols='Store', prediction_length=1)

        lagged = LaggedTarget(range(1, 4), guac.config, guac.logger)
        out = lagged.execute(guac.data)

        out.df = out.df.sort_values(['Store', 'Date'])
        self.assertTrue(np.isnan(out.df['Sales_lagged_1'].iloc[0]))
        self.assertAlmostEqual(out.df['Sales_lagged_1'].iloc[1], 0, delta=0.01)
        self.assertAlmostEqual(out.df['Sales_lagged_1'].iloc[2], 87104889, delta=0.01)
        self.assertAlmostEqual(out.df['Sales_lagged_2'].iloc[3], 87104889, delta=0.01)
