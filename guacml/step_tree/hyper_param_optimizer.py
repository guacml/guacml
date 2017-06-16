import pandas as pd
from hyperopt import Trials
from hyperopt import fmin, tpe


class HyperParameterOptimizer:

    def __init__(self, model_runner, features):
        self.model_runner = model_runner
        self.features = features

    def optimize(self, hyper_param_iterations, old_trials=None):
        hp_info = self.model_runner.hyper_parameter_info()
        if old_trials is None:
            trials = Trials()
        else:
            trials = old_trials

        best_hps = fmin(self.to_minimize,
                        hp_info,
                        algo=tpe.suggest,
                        max_evals=hyper_param_iterations,
                        trials=trials)

        filtered_hps = {}
        for hp in self.model_runner.hyper_parameter_info():
            try:
                filtered_hps[hp] = best_hps[hp]
            except KeyError:
                filtered_hps[hp] = None

        return trials, filtered_hps

    def to_minimize(self, args):
        return self.model_runner.train_and_cv_error(self.features, args)

    @staticmethod
    def trials_to_data_frame(trials):
        all_trials = []
        for trial in trials.trials:
            unpacked = {
                'cv error': trial['result']['loss'],
                'status': trial['result']['status'],
                'run': trial['misc']['tid']
            }

            param_vals = trial['misc']['vals']
            for key in param_vals:
                value_list = param_vals[key]
                if len(value_list) == 1:
                    unpacked[key] = value_list[0]
                elif len(value_list) == 0:
                    unpacked[key] = None
                else:
                    raise Exception('Unexpected number of hyper parameter results.')
            all_trials.append(unpacked)

        return pd.DataFrame(all_trials)
