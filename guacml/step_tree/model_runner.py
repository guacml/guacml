from sklearn.metrics import log_loss

from guacml.step_tree.hyper_param_optimizer import HyperParameterOptimizer
from guacml.step_tree.model_result import ModelResult
from .base_step import BaseStep


class ModelRunner(BaseStep):
    def __init__(self, model, target, hyper_param_iterations):
        self.model = model
        self.target = target
        self.hyper_param_iterations = hyper_param_iterations

    def execute(self, input, metadata):
        train, holdout = self.splitter.split(input)
        features = self.model.select_features(metadata)
        features = features[features != self.target]

        hp_optimizer = HyperParameterOptimizer(self.model, train, features,
                                               self.target, self.splitter)
        hyper_params, cv_error, all_hp_runs = hp_optimizer.optimize(self.hyper_param_iterations)

        training_error, _ = self.score_model(train, features)
        holdout_error, holdout_predictions = self.score_model(holdout, features)

        return ModelResult(self.model,
                           training_error,
                           cv_error,
                           holdout_error,
                           holdout_predictions,
                           hyper_params,
                           all_hp_runs)

    def score_model(self, input, features):
        predictions = self.model.predict(input[features])
        return log_loss(input[self.target], predictions), predictions