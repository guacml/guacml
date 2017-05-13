from guacml.preprocessing.column_analyzer import ColType
from ..preprocessing.base_step import BaseStep
from sklearn.ensemble import RandomForestClassifier

class RandomForest(BaseStep):
    def __init__(self, target):
        self.target = target

    def execute(self, input, metadata):
        classifier = RandomForestClassifier()
        valid_types = [ColType.BINARY, ColType.NUMERIC, ColType.ORDINAL, ColType.INT_ENCODING]
        valid_cols = metadata[(metadata.type.isin(valid_types)) &
                              (metadata.col_name != self.target)].col_name

        classifier.fit(input[valid_cols], input[self.target])

        return classifier
