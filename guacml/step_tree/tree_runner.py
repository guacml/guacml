from guacml.step_tree.model_manager import ModelManager
import os
import yaml
import json
import pandas as pd


# ToDo: Refactor the file handling into separate class
class TreeRunner:
    def __init__(self, data, config, tree):
        self.data = data
        self.config = config
        self.tree = tree
        self.min_hp_iterations = None
        self.all_prev_runs = None
        self.prev_run = None
        self.current_run = None

    def run(self, min_hyper_param_iterations):
        """
        Either runs all preprocessing steps or loads the preprocessed
        data from files, if they have been run before
        """
        self.min_hp_iterations = min_hyper_param_iterations
        self.load_previous_runs()

        if self.prev_run is None:
            results = {}
            data_version, config_version = self.get_next_version()
            self.all_prev_runs[data_version][config_version] = self.current_run = {}
            self.current_run['models'] = {}
            self.current_run['data_version'] = data_version
            self.current_run['config_version'] = config_version
            self.current_run['config'] = self.config
            self.run_step(self.tree.root_name, self.data, results)
            return results
        else:
            model_step_names = self.tree.get_leaf_names()
            for model_name in model_step_names:
                data_path = self.prev_run['model_data_paths'][model_name]
                data = pd.read_feather(data_path)
                model_step = self.tree.get_step(model_name)
                model_step.execute(data, self.min_hp_iterations)

    def run_step(self, step_name, data, results):
        print('Running step ' + step_name)
        children = self.tree.get_children(step_name)
        step = self.tree.get_step(step_name)

        if isinstance(step, ModelManager):
            results[step_name] = step.execute(data, self.min_hp_iterations)
            data_path = self.get_data_path()
            data.to_feather(data_path)
            self.current_run['model_data_paths'][step_name] = data_path
        else:
            dataset = step.execute(data)
            for child in children:
                self.run_step(child, dataset, results)

    def get_next_version(self):

    def get_data_path(self):

    def load_previous_runs(self):
        model_input_folder = './data/model_input'
        if not os.path.exists(model_input_folder):
            os.makedirs(model_input_folder)

        prev_runs_file = os.path.join(model_input_folder, 'previous_runs.yaml')
        if not os.path.isfile(prev_runs_file):
            return

        with open(prev_runs_file, 'r') as file:
            self.all_prev_runs = yaml.load(file)

            found_run_ids = []
            for data_version, config_versions in self.all_prev_runs.items():
                for config_version, details in config_versions.items():
                    if details.input_data_hash == self.data.df_hash and \
                       details.config_hash == self.config_hash():
                        found_run_ids.append((data_version, config_version))
            if len(found_run_ids) == 0:
                pass
            elif len(found_run_ids) == 1:
                data_version, config_version = found_run_ids[0]
                self.prev_run = self.all_prev_runs[data_version][config_version]
            else:
                raise Exception('Duplicate entries of previous runs found.')

    def config_hash(self):
        return hash(json.dumps(self.config, sort_keys=True))






