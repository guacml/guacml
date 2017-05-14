from sklearn.linear_model import LogisticRegression

from guacml.models.base_model import BaseModel
from guacml.preprocessing.column_analyzer import ColType


class LinearModel(BaseModel):
    def get_valid_types(self):
        return [ColType.BINARY, ColType.NUMERIC, ColType.ORDINAL]

    def get_adapter(self):
        return Adapter()

    def train(self, x, y):
        self.classifier = LogisticRegression()
        self.classifier.fit(x, y)

    def predict(self, x):
        return self.classifier.predict(x)


