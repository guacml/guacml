import guacml as guac
import os
import numpy as np
import unittest

from guacml.metrics.root_mean_squared_log_error import RootMeanSquaredLogError


class TestRootMeanSquaredLogError(unittest.TestCase):

    def test_correct_true(self):
        rsmle = RootMeanSquaredLogError()
        error = rsmle.error(np.ones(1), np.ones(1))
        self.assertAlmostEqual(0, error, delta=1e-15)

    def test_incorrect_true(self):
        rsmle = RootMeanSquaredLogError()
        error = rsmle.error(np.ones(1), np.zeros(1))
        self.assertAlmostEqual(0.6931471806, error, delta=1e-9)

    def test_incorrect_false(self):
        rsmle = RootMeanSquaredLogError()
        error = rsmle.error(np.zeros(1), np.ones(1))
        self.assertAlmostEqual(0.6931471806, error, delta=1e-9)

    def test_mix(self):
        rsmle = RootMeanSquaredLogError()
        error = rsmle.error(np.array([0,0,1,1]), np.array([0,1,0,1]))
        self.assertAlmostEqual(0.4901290717, error, delta=1e-9)

    def test_ignore_negative_prediction(self):
        rsmle = RootMeanSquaredLogError()
        error = rsmle.error(0, -1)
        self.assertAlmostEqual(0, error, delta=1e-15)
