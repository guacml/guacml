import unittest

from guacml import splitters
from tests.test_util import load_config


class TestSplitters(unittest.TestCase):
    def test_create_random_splitter(self):
        config = load_config()

        self.assertIsNone(config['run_time']['time_series']['date_split_col'])

        splitter = splitters.create(config)

        self.assertIsInstance(splitter, splitters.random_splitter.RandomSplitter)
