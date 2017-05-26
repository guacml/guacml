from guacml.step_tree.feature_reducer import FeatureReducer
from guacml.step_tree.hyper_param_optimizer import HyperParameterOptimizer
from guacml.step_tree.model_result import ModelResult

from guacml.step_tree.model_runner import ModelRunner
from .base_step import BaseStep


class ModelManager(BaseStep):
    def __init__(self, model, target, hyper_param_iterations, eval_metric):
        self.model = model
        self.target = target
        self.hyper_param_iterations = hyper_param_iterations
        self.eval_metric = eval_metric
        self.splitter = None


    def execute(self, input, metadata):
        model_runner = ModelRunner(self.model, input, self.target, self.eval_metric, self.splitter)
        features = self.select_features(metadata)
        features = features[features != self.target]

        hp_optimizer = HyperParameterOptimizer(model_runner, features)
        all_trials, best_hps = hp_optimizer.optimize(self.hyper_param_iterations)

        feature_reducer = FeatureReducer(model_runner, best_hps)
        features = feature_reducer.reduce(features)

        return self.build_result(model_runner, metadata, features, all_trials, best_hps)

    def select_features(self, metadata):
        return metadata[metadata.type.isin(self.model.get_valid_types())].col_name

    def build_result(self, model_runner, metadata, features, all_trials, best_hps):
        all_trials = all_trials.sort_values('cv error')
        best = all_trials.iloc[0]

        model_runner.train_final_model(features, best_hps)
        training_error = model_runner.training_error()
        holdout_predictions = model_runner.holdout_predictions()
        holdout_error = model_runner.holdout_error()
        holdout_error_interval = model_runner.holdout_error_interval()
        holdout_row_errors = model_runner.row_wise_holdout_error()

        holdout = model_runner.holdout.copy()
        holdout['prediction'] = holdout_predictions
        holdout['error'] = holdout_row_errors

        return ModelResult(self.model,
                           features,
                           self.target,
                           training_error,
                           best['cv error'],
                           holdout_error,
                           holdout_error_interval,
                           holdout,
                           metadata,
                           best,
                           all_trials)
