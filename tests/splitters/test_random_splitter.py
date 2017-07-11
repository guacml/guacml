import unittest
import numpy as np
import pandas as pd

from guacml.splitters.random_splitter import RandomSplitter


class TestRandomSplitter(unittest.TestCase):

    def test_single_split(self):
        df = pd.DataFrame({'a': np.ones(1000)})
        splitter = RandomSplitter({'n_folds': 1, 'holdout_size': 0.001, 'cv_size': 0.001})
        train_and_cv, holdout = splitter.holdout_split(df)
        train_cv_splits = splitter.cv_splits(train_and_cv)

        self.assertEqual(1, holdout.shape[0])
        self.assertEqual(1, len(train_cv_splits))
        train, cv = train_cv_splits[0]
        self.assertEqual(998, train.shape[0])
        self.assertEqual(1, cv.shape[0])
