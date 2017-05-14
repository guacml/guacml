from guacml.models.base_model import BaseModel, BaseAdapter
from guacml.preprocessing.column_analyzer import ColType
from sklearn.ensemble import RandomForestClassifier


class RandomForest(BaseModel):
    def get_valid_types(self):
        return [ColType.BINARY, ColType.NUMERIC, ColType.ORDINAL, ColType.INT_ENCODING]

    def get_adapter(self):
        return Adapter()


class Adapter(BaseAdapter):
    def train(self, x, y):
        self.classifier = RandomForestClassifier()
        self.classifier.fit(x, y)

    def predict(self, x):
        return self.classifier.predict(x)
