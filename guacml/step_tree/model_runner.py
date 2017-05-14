from sklearn.metrics import log_loss

from guacml.step_tree.model_result import ModelResult
from .base_step import BaseStep


class ModelRunner(BaseStep):
    def __init__(self, model, target):
        self.model = model
        self.target = target

    def execute(self, input, metadata):
        train, cv = self.splitter.split(input)
        features = self.model.select_features(metadata)
        features = features[features != self.target]

        model = self.model.train(train[features], train[self.target])
        train_predictions = self.model.predict(train[features])
        training_error = log_loss(train[self.target], train_predictions)
        cv_predictions = self.model.predict(cv[features])
        cv_error = log_loss(cv[self.target], cv_predictions)

        return ModelResult(model, training_error, cv_error, cv_predictions)

