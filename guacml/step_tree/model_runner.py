from sklearn.metrics import log_loss

from guacml.step_tree.hyper_param_optimizer import HyperParameterOptimizer
from guacml.step_tree.model_result import ModelResult
from .base_step import BaseStep
import numpy as np

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
        holdout_row_errors = self.rowise_log_loss(holdout[self.target], holdout_predictions)

        return ModelResult(self.model,
                           training_error,
                           best['cv error'],
                           holdout_error,
                           holdout_predictions,
                           holdout_row_errors,
                           metadata,
                           best,
                           all_trials)

    def score_model(self, input, features):
        predictions = self.model.predict(input[features])
        return log_loss(input[self.target], predictions), predictions

    @staticmethod
    def rowise_log_loss(truth, prediction):
        eps = 1e-15
        prediction = np.clip(prediction, eps, 1 - eps)
        return -1 * (truth * np.log(prediction) + (1 - truth) * np.log(1 - prediction))