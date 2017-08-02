# ToDo: Deprecation warning because xgboost import cross_validation
import xgboost as xgb
import numpy as np
import pandas as pd

from guacml.models.base_model import BaseModel
from hyperopt import hp


class XGBoost(BaseModel):

    def __init__(self, config, logger):
        super().__init__(config, logger)
        self.model_config = config['models']['xgboost']

    def copy_model(self):
        booster = xgb.Booster()
        booster.load_model(self.model.save_raw())

        return booster

    def get_valid_types(self):
        return ['binary', 'numeric', 'ordinal', 'int_encoding']

    def hyper_parameter_info(self):
        if 'hyper_parameters' in self.model_config:
            hps = self.model_config['hyper_parameters'].copy()
            hps['fixed'] = True
            return hps

        return {
            'n_rounds': hp.qlognormal('n_rounds', 4, 1, 1),
            'max_depth': hp.qlognormal('max_depth', 1.6, 0.3, 1)
        }

    def train(self, x, y, n_rounds=100, max_depth=5):
        n_rounds = self.pos_int(n_rounds)
        max_depth = int(max_depth)
        dtrain = xgb.DMatrix(x, y, missing=np.nan)
        params = {
            'booster': 'gbtree',
            'eta': 0.2,
            'silent': True,
            'max_depth': self.pos_int(max_depth)
        }

        params.update(self.model_config)

        if 'objective' not in params:
            if self.problem_type == 'binary_clas':
                params['objective'] = 'reg:logistic'
            elif self.problem_type == 'regression':
                params['objective'] = 'reg:linear'
            else:
                raise NotImplementedError(
                    'Problem type {0} not implemented for XGBoost.'.format(self.problem_type)
                )

        self.logger.info('About to train %d iterations of xgboost using %s', n_rounds, params)
        self.model = xgb.train(params, dtrain, n_rounds)

    def predict(self, x):
        dfeatures = xgb.DMatrix(x, missing=np.nan)
        prediction = self.model.predict(dfeatures)
        return pd.Series(prediction, index=x.index)

    def feature_importances(self, x):
        feat_scores = self.model.get_fscore()
        return pd.Series(list(feat_scores.values()), index=list(feat_scores.keys()))

    def get_state(self):
        return self.model.save_raw()
