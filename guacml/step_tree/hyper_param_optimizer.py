import pandas as pd
from sklearn.metrics import log_loss
from bayes_opt import BayesianOptimization


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
        hp_ranges = {param: info.range for param, info in hp_info.items()}

        first_item = next(iter(hp_info.items()))
        n_init_points = len(first_item[1].init_points)
        if n_init_points <= hyper_param_iterations:
            init_points = {hp: info.init_points for hp, info in hp_info.items()}
        else:
            init_points = {hp: info.init_points[:hyper_param_iterations] for hp, info in hp_info.items()}
        n_iter = max(hyper_param_iterations - n_init_points, 1)

        bayes_opt = BayesianOptimization(self.to_maximize, hp_ranges, verbose=0)
        bayes_opt.explore(init_points)
        bayes_opt.maximize(init_points=0, n_iter=n_iter)

        all_runs = pd.DataFrame(bayes_opt.res['all']['params'])
        all_runs['cv error'] = bayes_opt.res['all']['values']

        max_cv_err = -bayes_opt.res['max']['max_val']
        max_params = bayes_opt.res['max']['max_params']

        return max_params, max_cv_err, all_runs

    def to_maximize(self, **kwargs):
        self.model.train(self.X_train, self.y_train, **kwargs)
        cv_predictions = self.model.predict(self.X_cv)
        cv_error = log_loss(self.y_cv, cv_predictions)
        return -cv_error
