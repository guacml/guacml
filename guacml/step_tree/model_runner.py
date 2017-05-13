from sklearn.metrics import log_loss
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

        result = Result()
        result.adapter = adapter
        result.training_error = log_loss(adapter.predict(train[features]), train[self.target])
        result.cv_error = log_loss(adapter.predict(cv[features]), cv[self.target])

        return result

class Result:
    pass
