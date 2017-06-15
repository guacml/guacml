import shutil
import os
import yaml
import joblib


class PreviousRuns():
    def __init__(self, data, config,
                 previous_runs_folder='./data/previous_runs'):
        self.found_matching_run = None
        self.config_ = config
        self.config_hash_ = joblib.hash(config)
        self.data_ = data
        self.run_ = None
        self.all_prev_runs_ = None
        self.max_data_version_ = 0
        self.max_config_version_ = 0

        self.previous_runs_folder = previous_runs_folder
        if not os.path.exists(previous_runs_folder):
            os.makedirs(previous_runs_folder)
        self.prev_runs_file_ = os.path.join(previous_runs_folder, 'previous_runs.yaml')

        self.load_previous_runs()

        if not self.found_matching_run:
            self.data_folder_ = self.get_versioned_folder(self.max_data_version_ + 1,
                                                         self.max_config_version_ + 1)
            if not os.path.exists(self.data_folder_):
                os.makedirs(self.data_folder_)
            with open(os.path.join(self.data_folder_, 'config.yaml'), 'w') as config_file:
                config_file.write(yaml.dump(self.config_, default_flow_style=False))
            self.create_run_()

    def get_versioned_folder(self, max_data_version, max_config_version):
        return os.path.join(self.previous_runs_folder,
                            'data_v_' + str(max_data_version),
                            'config_v_' + str(max_config_version))

    def load_previous_runs(self):
        if not os.path.isfile(self.prev_runs_file_):
            self.found_matching_run = False
            return

        with open(self.prev_runs_file_, 'r') as file:
            self.all_prev_runs_ = yaml.load(file)
            for run in self.all_prev_runs_:
                self.max_data_version_ = max(self.max_data_version_, run['input_data_version'])
                if run['input_data_hash'] == self.data_.df_hash:
                    self.max_config_version_ = max(self.max_config_version_, run['config_version'])
                    if run['config_hash'] == self.config_hash_:
                        if self.found_matching_run:
                            raise Exception('Duplicate previous run entries found.')
                        self.found_matching_run = True
                        self.run_ = run
                        self.data_folder_ = self.get_versioned_folder(run['input_data_version'],
                                                                      run['config_version'])

        if self.found_matching_run is None:
            self.found_matching_run = False

    # def get_model_input(self, model_name):
    #     data_path = self.run_['model_data_paths'][model_name]
    #     return pd.read_feather(data_path)

    def get_prev_results(self):
        model_results = self.run_['model_result_paths']
        return {name: joblib.load(path) for name, path in model_results.items()}

    def create_run_(self):
        self.run_ = {
            'input_data_version': self.max_data_version_ + 1,
            'input_data_hash': self.data_.df_hash,
            'config_version': self.max_config_version_ + 1,
            'config_hash': self.config_hash_,
            #'config': self.config_,
            #'model_data_paths': {},
            'model_result_paths': {}
        }

    # def add_model_input(self, model_name, data):
    #     data_path = os.path.join(self.data_folder_, model_name + '_input.feather')
    #     data.df.to_feather(data_path)
    #     self.run_['model_data_paths'][model_name] = data_path

    def add_model_result(self, model_name, result):
        result_path = os.path.join(self.data_folder_, model_name + '_result.joblib_dump.gz')
        joblib.dump(result, result_path)
        self.run_['model_result_paths'][model_name] = result_path

    def store_run(self):
        print(self.found_matching_run)
        if not self.found_matching_run:
            with open(self.prev_runs_file_, 'w') as file:
                if self.all_prev_runs_ is None:
                    self.all_prev_runs_ = [self.run_]
                else:
                    self.all_prev_runs_.append(self.run_)
                file.write(yaml.dump(self.all_prev_runs_, default_flow_style=False))

    def clear(self):
        shutil.rmtree(self.previous_runs_folder)

