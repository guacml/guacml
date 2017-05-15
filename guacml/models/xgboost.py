# ToDo: Deprecation warning because xgboost import cross_validation
import xgboost as xgb
import numpy as np
from guacml.models.base_model import BaseModel
from guacml.models.hyper_param_info import HyperParameterInfo
from guacml.preprocessing.column_analyzer import ColType

N_ROUNDS_DEFAULT = 50
MAX_DEPTH_DEFAULT = 4


class XgBoost(BaseModel):
    def get_valid_types(self):
        return [ColType.BINARY, ColType.NUMERIC, ColType.ORDINAL, ColType.INT_ENCODING]

    @staticmethod
    def hyper_parameter_info():
        return {
            'n_rounds': HyperParameterInfo(N_ROUNDS_DEFAULT,
                                           [10, 1000],
                                           [10, 100, 500]),
            'max_depth': HyperParameterInfo(MAX_DEPTH_DEFAULT,
                                            [3, 10],
                                            [7, 5, 3])
        }

    def train(self, x, y, n_rounds=N_ROUNDS_DEFAULT, max_depth=MAX_DEPTH_DEFAULT):
        dtrain = xgb.DMatrix(x, y, missing=np.nan)
        params = {
            "objective": "reg:logistic",
            "booster" : "gbtree",
            "eta": 0.2,
            "max_depth": self.pos_int(max_depth)
        }
        self.xgb_model = xgb.train(params, dtrain, self.pos_int(n_rounds))

    def predict(self, x):
        dfeatures = xgb.DMatrix(x, missing=np.nan)
        return self.xgb_model.predict(dfeatures)

    @staticmethod
    def pos_int(value):
        return max(int(value), 0)