import unittest
import pandas as pd
import logging

from tests.test_util import load_config
from guacml.dataset import Dataset
from guacml.preprocessing.feature_whitelist import FeatureWhitelist


class TestFeatureWhitelist(unittest.TestCase):

    def setUp(self):
        self.config = load_config()
        self.config['run_time']['target'] = 'tt'
        self.logger = logging.getLogger(__name__)
        data = {'aa': [1, 2, 3], 'bb': [4, 5, 6], 'tt': [7, 8, 9]}
        self.data = Dataset.from_df(pd.DataFrame(data), self.config,
                                    self.config['run_time']['target'], None, self.logger)

    def test_whitelisting_disabled(self):
        step = FeatureWhitelist(self.config, self.logger)
        step.execute_inplace(self.data)

        self.assertEqual(['aa', 'bb', 'tt'], self.data.df.columns.tolist())
        self.assertEqual(['aa', 'bb', 'tt'], self.data.metadata.index.tolist())

    def test_whitelisting_enabled(self):
        self.config['pre_processing']['feature_whitelist'] = ['aa']
        step = FeatureWhitelist(self.config, self.logger)
        step.execute_inplace(self.data)

        self.assertEqual(['aa', 'tt'], self.data.df.columns.tolist())
        self.assertEqual(['aa', 'tt'], self.data.metadata.index.tolist())
