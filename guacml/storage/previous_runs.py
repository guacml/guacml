import pandas as pd
import os
import yaml
import json


class PreviousRuns():
    def __init__(self, data, config):
        self.config = self.config
        self.config_hash = hash(json.dumps(config, sort_keys=True))
        self.data_hash = data.df_hash

        self.run = None
        self.exists = None
        self.max_data_version = 0
        self.max_config_version = 0

        previous_runs_folder = './data/previous_runs'
        if not os.path.exists(previous_runs_folder):
            os.makedirs(previous_runs_folder)
        self.prev_runs_file = os.path.join(previous_runs_folder, 'previous_runs.yaml')

        self.load_previous_runs()

        if not self.exists:
            self.data_folder = os.path.join(previous_runs_folder,
                                       'data_v_' + str(self.max_data_version + 1),
                                       'config_v_' + str(self.max_config_version + 1))
            if not os.path.exists(self.data_folder):
                os.makedirs(self.data_folder)
            self.create_run_()

    def load_previous_runs(self):

        if not os.path.isfile(self.prev_runs_file):
            self.exists = False
            return

        with open(self.prev_runs_file, 'r') as file:
            all_prev_runs = yaml.load(file)
            for run in all_prev_runs:
                self.max_data_version = max(self.max_data_version, run['data_version'])
                if run['input_data_hash'] == self.data_hash:
                    self.data_found = True
                    self.max_config_version = max(self.max_config_version, run['config_version'])
                    if run['config_hash'] == self.config_hash:
                        if self.exists:
                            raise Exception('Duplicate previous run entries found.')
                        self.exists = True
                        self.run = run
        if self.exists is None:
            self.exists = False

    def get_model_input(self, model_name):
        data_path = self.run['model_data_paths'][model_name]
        return pd.read_feather(data_path)

    def create_run_(self):
        self.run = {
            'models': {},
            'data_version': self.max_data_version + 1,
            'config_version': self.max_config_version + 1,
            'config': self.config
        }

    def add_model_input(self, model_name, data):
        data_path = os.path.join(self.data_folder, model_name + '.feather')
        data.to_feather(data_path)
        self.run['model_data_paths'][model_name] = data_path

    def store_run(self):
        if self.exists:
            raise Exception('Tried to store data although a previous run was found.')
        with open(self.prev_runs_file, 'a') as file:
            file.write(yaml.dump(self.run))

