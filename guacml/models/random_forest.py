from ..preprocessing.base_step import BaseStep
from sklearn.ensemble import RandomForestClassifier

class RandomForest(BaseStep):
    def __init__(self, target):
        self.target = target

    def execute(self, input):
        classifier = RandomForestClassifier()
        features = input.columns
        features.remove(self.target)
        classifier.train(input[features], input[self.target])
