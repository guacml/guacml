# ToDo: Deprecation warning because sgboost import cross_validation
import xgboost as xgb
import numpy as np
from guacml.models.base_model import BaseModel
from guacml.preprocessing.column_analyzer import ColType


class XgBoost(BaseModel):
    def get_valid_types(self):
        return [ColType.BINARY, ColType.NUMERIC, ColType.ORDINAL, ColType.INT_ENCODING]

    def train(self, x, y):
        dtrain = xgb.DMatrix(x, y, missing=np.nan)
        params = {
            "objective": "reg:logistic",
            "booster" : "gbtree",
            "eta": 0.2,
            "max_depth": 5
        }
        self.xgb_model = xgb.train(params, dtrain, 50)

    def predict(self, x):
        dfeatures = xgb.DMatrix(x, missing=np.nan)
        return self.xgb_model.predict(dfeatures)