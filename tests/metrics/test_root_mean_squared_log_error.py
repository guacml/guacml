import numpy as np
import unittest

from guacml.metrics.root_mean_squared_log_error import RootMeanSquaredLogError


class TestRootMeanSquaredLogError(unittest.TestCase):

    def test_one_one(self):
        rsmle = RootMeanSquaredLogError()
        error = rsmle.error(np.ones(1), np.ones(1))
        self.assertAlmostEqual(0, error, delta=1e-15)

    def test_one_two(self):
        rsmle = RootMeanSquaredLogError()
        error = rsmle.error(np.ones(1), np.array([2]))
        self.assertAlmostEqual(0.6931471806, error, delta=1e-9)

    def test_mix(self):
        rsmle = RootMeanSquaredLogError()
        error = rsmle.error(np.array([0.1, 0.1, 1, 1]), np.array([0.11, 1, 0.1, 1]))
        self.assertAlmostEqual(1.628870793, error, delta=1e-9)

    def test_raise_for_negative_prediction(self):
        rsmle = RootMeanSquaredLogError()
        with self.assertRaises(ValueError):
            rsmle.error(np.ones(1), -np.ones(1))

