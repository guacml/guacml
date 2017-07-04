# ToDo: Deprecation warning because xgboost import cross_validation
import xgboost as xgb
import numpy as np
import pandas as pd

from guacml.enums import ProblemType
from guacml.models.base_model import BaseModel
from guacml.preprocessing.column_analyzer import ColType
from hyperopt import hp


class XgBoost(BaseModel):
    def __init__(self, problem_type, config=None):
        super().__init__(problem_type)
        self.config = config

    def get_valid_types(self):
        return [ColType.BINARY, ColType.NUMERIC, ColType.ORDINAL, ColType.INT_ENCODING]

    @staticmethod
    def hyper_parameter_info():
        return {
            'n_rounds': hp.qlognormal('n_rounds', 4, 1, 1),
            'max_depth': hp.qlognormal('max_depth', 1.6, 0.3, 1)
        }

    def train(self, x, y, n_rounds=100, max_depth=5):
        n_rounds = int(n_rounds)
        max_depth = int(max_depth)
        dtrain = xgb.DMatrix(x, y, missing=np.nan)
        params = {
            'booster': 'gbtree',
            'eta': 0.2,
            'silent': True,
            'max_depth': self.pos_int(max_depth)
        }

        if self.config is not None:
            params.update(self.config)

        if 'objective' not in params:
            if self.problem_type == ProblemType.BINARY_CLAS:
                params['objective'] = 'reg:logistic'
            elif self.problem_type == ProblemType.REGRESSION:
                params['objective'] = 'reg:linear'
            else:
                raise NotImplementedError(
                    'Problem type {0} not implemented for XgBoost.'.format(self.problem_type)
                )

        self.model = xgb.train(params, dtrain, self.pos_int(n_rounds))

    def predict(self, x):
        dfeatures = xgb.DMatrix(x, missing=np.nan)
        prediction = self.model.predict(dfeatures)
        return pd.Series(prediction, index=x.index)

    def feature_importances(self, x):
        feat_scores = self.model.get_fscore()
        return pd.Series(list(feat_scores.values()), index=list(feat_scores.keys()))
