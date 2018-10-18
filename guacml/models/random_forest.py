import pandas as pd
import pickle
from hyperopt import hp
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier

from guacml.models.base_model import BaseModel


class RandomForest(BaseModel):

    def copy_model(self):
        return pickle.loads(pickle.dumps(self.model))

    def get_valid_types(self):
        return ['binary', 'numeric', 'ordinal', 'int_encoding']

    def hyper_parameter_info(self):
        return {
            'n_estimators': hp.qlognormal('n_estimators', 4, 1, 1),
            'max_depth': hp.choice('use_max_depth',
                                   [None, hp.qlognormal('max_depth', 3, 1, 1)]),
            'min_samples_leaf': hp.qlognormal('min_samples_leaf', 2, 1, 1) + 1
        }

    def train(self, x, y,
              n_estimators=10,
              max_depth=None,
              min_samples_leaf=1):
        n_estimators = self.to_int(n_estimators)
        max_depth = self.to_int(max_depth)
        min_samples_leaf = self.pos_int(min_samples_leaf)

        if self.problem_type == 'binary_clas':
            self.model = RandomForestClassifier(n_estimators,
                                                max_depth=max_depth,
                                                min_samples_leaf=min_samples_leaf)
        elif self.problem_type == 'regression':
            self.model = RandomForestRegressor(n_estimators,
                                               max_depth=max_depth,
                                               min_samples_leaf=min_samples_leaf)
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
        feat_scores = self.model.feature_importances_
        return pd.Series(list(feat_scores), index=list(x.columns))

    def get_state(self):
        return pickle.dumps(self.model)

    def set_state(self, state):
        self.model = pickle.loads(state)
