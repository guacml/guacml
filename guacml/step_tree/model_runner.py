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
        all_trials = hp_optimizer.optimize(self.hyper_param_iterations)
        all_trials = all_trials.sort_values('cv error')
        best = all_trials.iloc[0]

        training_error, _ = self.score_model(train, features)
        holdout_error, holdout_predictions = self.score_model(holdout, features)

        return ModelResult(self.model,
                           training_error,
                           best['cv error'],
                           holdout_error,
                           holdout_predictions,
                           best,
                           all_trials)

    def score_model(self, input, features):
        predictions = self.model.predict(input[features])
        return log_loss(input[self.target], predictions), predictions