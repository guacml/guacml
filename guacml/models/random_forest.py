from guacml.preprocessing.column_analyzer import ColType
from sklearn.ensemble import RandomForestClassifier

# TODO base model class
class RandomForest:
    def select_features(self, metadata):
        valid_types = [ColType.BINARY, ColType.NUMERIC, ColType.ORDINAL, ColType.INT_ENCODING]

        return metadata[metadata.type.isin(valid_types)].col_name

    def get_adapter(self):
        return Adapter()

class Adapter:
    def train(self, x, y):
        self.classifier = RandomForestClassifier()
        self.classifier.fit(x, y)

    def predict(self, x):
        return self.classifier.predict(x)
