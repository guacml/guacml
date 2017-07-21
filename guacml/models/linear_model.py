from sklearn.linear_model import LogisticRegression
from hyperopt import hp
from sklearn.linear_model import Ridge

from guacml.models.base_model import BaseModel
from guacml.preprocessing.column_analyzer import ColType
import pandas as pd
import numpy as np


class LinearModel(BaseModel):

    def get_valid_types(self):
        return [ColType.BINARY, ColType.NUMERIC, ColType.ORDINAL, ColType.INT_ENCODING]

    def hyper_parameter_info(self):
        return {
            'alpha': hp.loguniform('alpha', -12, 3)
        }

    def train(self, x, y, alpha=1):
        if self.problem_type == 'binary_clas':
            self.model = LogisticRegression(C=alpha)
        elif self.problem_type == 'regression':
            self.model = Ridge(alpha=alpha)
        else:
            raise NotImplementedError('Problem type {0} not implemented'.format(self.problem_type))

        self.model.fit(x, y)

    def predict(self, x):
        if self.problem_type == 'binary_clas':
            prediction = self.model.predict_proba(x)[:, 1]
        elif self.problem_type == 'regression':
            prediction = self.model.predict(x)
        else:
            raise NotImplementedError('Problem type {0} not implemented'.format(self.problem_type))

        return pd.Series(prediction, index=x.index)

    def feature_importances(self, x):
        importance = np.abs(self.model.coef_) * x.std().values
        return pd.Series(importance.flatten(), index=x.columns)
