import pandas as pd
from hyperopt import Trials
from sklearn.metrics import log_loss
from hyperopt import fmin, tpe


class HyperParameterOptimizer:

    def __init__(self, model, input, features, target, splitter):
        self.model = model
        train, cv = splitter.split(input)
        self.X_train = train[features]
        self.y_train = train[target]
        self.X_cv = cv[features]
        self.y_cv = cv[target]

    def optimize(self, hyper_param_iterations):
        hp_info = self.model.hyper_parameter_info()

        # first_item = next(iter(hp_info.items()))
        # n_init_points = len(first_item[1].init_points)
        # if n_init_points <= hyper_param_iterations:
        #     init_points = {hp: info.init_points for hp, info in hp_info.items()}
        # else:
        #     init_points = {hp: info.init_points[:hyper_param_iterations] for hp, info in hp_info.items()}
        # n_iter = max(hyper_param_iterations - n_init_points, 1)

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
        cv_error = log_loss(self.y_cv, cv_predictions)
        return cv_error
