from sklearn.linear_model import LogisticRegression
from hyperopt import hp

from guacml.models.base_model import BaseModel
from guacml.models.hyper_param_info import HyperParameterInfo
from guacml.preprocessing.column_analyzer import ColType


class LinearModel(BaseModel):
    def get_valid_types(self):
        return [ColType.BINARY, ColType.NUMERIC, ColType.ORDINAL]

    @staticmethod
    def hyper_parameter_info():
        return HyperParameterInfo({
            'C': hp.loguniform('C', -12, 3)
        })

    def train(self, x, y, C=1):
        self.lin_model = LogisticRegression(C=C)
        self.lin_model.fit(x, y)

    def predict(self, x):
        return self.lin_model.predict_proba(x)[:, 1]


