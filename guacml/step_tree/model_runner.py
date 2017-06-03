from guacml.step_tree.hyper_param_optimizer import HyperParameterOptimizer
from guacml.step_tree.model_result import ModelResult
from .base_step import BaseStep


class ModelRunner(BaseStep):
    def __init__(self, model, target, hyper_param_iterations, eval_metric):
        self.model = model
        self.target = target
        self.hyper_param_iterations = hyper_param_iterations
        self.eval_metric = eval_metric

    def execute(self, dataframe, metadata):
        train_and_cv, holdout = self.splitter.split(dataframe)
        train, cv = self.splitter.split(train_and_cv)
        features = self.model.select_features(metadata)
        features = features[features != self.target]

        hp_optimizer = HyperParameterOptimizer(self.model, train, cv, features,
                                               self.target, self.eval_metric)
        all_trials = hp_optimizer.optimize(self.hyper_param_iterations)
        all_trials = all_trials.sort_values('cv error')
        best = all_trials.iloc[0]

        training_error, _ = self.score_model(train, features)
        holdout_error, holdout_predictions = self.score_model(holdout, features)
        holdout_row_errors = self.eval_metric.row_wise_error(holdout[self.target],
                                                             holdout_predictions)

        holdout = holdout.copy()
        holdout['error'] = holdout_row_errors
        holdout['prediction'] = holdout_predictions

        return ModelResult(self.model,
                           self.target,
                           training_error,
                           best['cv error'],
                           holdout_error,
                           holdout,
                           metadata,
                           best,
                           all_trials)

    def score_model(self, dataframe, features):
        predictions = self.model.predict(dataframe[features])
        return self.eval_metric.error(dataframe[self.target], predictions), predictions
