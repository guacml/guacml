import pandas as pd
from hyperopt import Trials
from hyperopt import fmin, tpe


class HyperParameterOptimizer:

    def __init__(self, model, train, cv, features, target, eval_metric):
        self.model = model
        self.X_train = train[features]
        self.y_train = train[target]
        self.X_cv = cv[features]
        self.y_cv = cv[target]
        self.eval_metric = eval_metric

    def optimize(self, hyper_param_iterations):
        hp_info = self.model.hyper_parameter_info()

        trials = Trials()
        fmin(self.to_minimize,
             hp_info.search_space,
             algo=tpe.suggest,
             max_evals=hyper_param_iterations,
             trials=trials)

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

    def to_minimize(self, args):
        self.model.train(self.X_train, self.y_train, **args)
        cv_predictions = self.model.predict(self.X_cv)
        cv_error = self.eval_metric.error(self.y_cv, cv_predictions)
        return cv_error
