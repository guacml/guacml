from sklearn.linear_model import LogisticRegression
from hyperopt import hp
from sklearn.linear_model import Ridge

from guacml.enums import ProblemType
from guacml.models.base_model import BaseModel
from guacml.models.hyper_param_info import HyperParameterInfo
from guacml.preprocessing.column_analyzer import ColType


class LinearModel(BaseModel):
    def get_valid_types(self):
        return [ColType.BINARY, ColType.NUMERIC, ColType.ORDINAL]

    @staticmethod
    def hyper_parameter_info():
        return HyperParameterInfo({
            'alpha': hp.loguniform('alpha', -12, 3)
        })

    def train(self, x, y, alpha=1):
        if self.problem_type == ProblemType.BINARY_CLAS:
            self.lin_model = LogisticRegression(C=alpha)
        elif self.problem_type == ProblemType.REGRESSION:
            self.lin_model = Ridge(alpha=alpha)
        else:
            raise NotImplementedError('Problem type {0} not implemented'.format(self.problem_type))

        self.lin_model.fit(x, y)

    def predict(self, x):
        if self.problem_type == ProblemType.BINARY_CLAS:
            return self.lin_model.predict_proba(x)[:, 1]
        elif self.problem_type == ProblemType.REGRESSION:
            return self.lin_model.predict(x)
        else:
            raise NotImplementedError('Problem type {0} not implemented'.format(self.problem_type))




