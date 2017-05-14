from sklearn.linear_model import LogisticRegression

from guacml.models.base_model import BaseModel
from guacml.models.hyper_param_info import HyperParameterInfo
from guacml.preprocessing.column_analyzer import ColType

C_DEFAULT = 1


class LinearModel(BaseModel):
    def get_valid_types(self):
        return [ColType.BINARY, ColType.NUMERIC, ColType.ORDINAL]

    @staticmethod
    def hyper_parameter_info():
        return {
            'C': HyperParameterInfo(C_DEFAULT, [10-6, 10])
        }

    def train(self, x, y, C=C_DEFAULT):
        self.lin_model = LogisticRegression(C=C)
        self.lin_model.fit(x, y)

    def predict(self, x):
        return self.lin_model.predict(x)


