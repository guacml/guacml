import unittest
import pandas as pd
import joblib
import shutil
import os
import yaml

import tests.test_util as test_util
from guacml.storage.previous_runs import PreviousRuns
from guacml.dataset import Dataset

# danger: this folder gets deleted after the tests. only change to folders that may be deleted.
PREV_RUN_FOLDER = './data/test_storage_data/previous_runs'
PREV_RUNS_FILE = os.path.join(PREV_RUN_FOLDER, 'previous_runs.yaml')


class TestPreviousRuns(unittest.TestCase):

    def tearDown(self):
        shutil.rmtree(PREV_RUN_FOLDER)

    @staticmethod
    def load_prev_runs_yml():
        with open(PREV_RUNS_FILE, 'r') as file:
            return yaml.load(file)

    def test_store_and_load(self):
        df = pd.DataFrame({'a':[0, 1], 'b': [2, 3]})
        data = Dataset(df, None, df_hash=joblib.hash(df))
        config = test_util.load_config()

        prev_runs = PreviousRuns(data, config, previous_runs_folder=PREV_RUN_FOLDER)
        data1 = Dataset(pd.DataFrame({'a': [1]}), None)
        data2 = Dataset(pd.DataFrame({'a': [2]}), None)

        prev_runs.add_model_result('result_1', pd.DataFrame({'a': [1]}))
        prev_runs.add_model_result('result_2', pd.DataFrame({'a': [2]}))
        prev_runs.store_run()

        prev_runs_2 = PreviousRuns(data, config, previous_runs_folder=PREV_RUN_FOLDER)

        model_results = prev_runs_2.get_prev_results()
        self.assertEqual(len(model_results), 2)
        self.assertEqual(model_results['result_2']['a'].iloc[0], 2)

    def test_store_and_new_data(self):
        df = pd.DataFrame({'a':[0, 1], 'b': [2, 3]})
        data = Dataset(df, None, df_hash=joblib.hash(df))
        config = test_util.load_config()

        prev_runs = PreviousRuns(data, config, previous_runs_folder=PREV_RUN_FOLDER)
        prev_runs.store_run()

        df2 = pd.DataFrame({'a':[1, 1], 'b': [2, 3]})
        data2 = Dataset(df2, None, df_hash=joblib.hash(df2))

        prev_runs_2 = PreviousRuns(data2, config, previous_runs_folder=PREV_RUN_FOLDER)
        prev_runs_2.store_run()

        prev_runs = self.load_prev_runs_yml()
        self.assertEqual(len(prev_runs), 2)

    def test_store_and_new_config(self):
        df = pd.DataFrame({'a':[0, 1], 'b': [2, 3]})
        data = Dataset(df, None, df_hash=joblib.hash(df))
        config = test_util.load_config()
        prev_runs = PreviousRuns(data, config, previous_runs_folder=PREV_RUN_FOLDER)
        prev_runs.store_run()

        config2 = test_util.load_config()
        config2['altered'] = True

        prev_runs_2 = PreviousRuns(data, config2, previous_runs_folder=PREV_RUN_FOLDER)
        prev_runs_2.store_run()

        prev_runs = self.load_prev_runs_yml()
        self.assertEqual(len(prev_runs), 2)

