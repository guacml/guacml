import numpy as np
import unittest

from guacml.metrics.log_loss import LogLoss


class TestLogLoss(unittest.TestCase):

    def test_correct_true(self):
        logloss = LogLoss()
        loss = logloss.error(np.ones(1), np.ones(1))
        self.assertAlmostEqual(0, loss, delta=1e-15)

    def test_incorrect_true(self):
        logloss = LogLoss()
        loss = logloss.error(np.ones(1), np.zeros(1))
        self.assertAlmostEqual(34.5387763949, loss, delta=1e-9)

    def test_incorrect_false(self):
        logloss = LogLoss()
        loss = logloss.error(np.zeros(1), np.ones(1))
        self.assertAlmostEqual(34.5395759923, loss, delta=1e-9)

    def test_mix(self):
        logloss = LogLoss()
        loss = logloss.error(np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1]))
        self.assertAlmostEqual(17.269588096, loss, delta=1e-9)
