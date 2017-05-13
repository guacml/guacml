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
        adapter = self.model.get_adapter()

        model = adapter.train(train[features], train[self.target])
        training_error = log_loss(adapter.predict(train[features]), train[self.target])
        cv_predictions = adapter.predict(cv[features])
        cv_error = log_loss(cv[self.target], cv_predictions)

        return ModelResult(model, training_error, cv_error, cv_predictions)

