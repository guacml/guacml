from guacml.models.base_model import BaseModel
from guacml.models.hyper_param_info import HyperParameterInfo
from guacml.preprocessing.column_analyzer import ColType
from sklearn.ensemble import RandomForestClassifier
from hyperopt import hp


class RandomForest(BaseModel):
    def get_valid_types(self):
        return [ColType.BINARY, ColType.NUMERIC, ColType.ORDINAL, ColType.INT_ENCODING]

    @staticmethod
    def hyper_parameter_info():
        return HyperParameterInfo({
            'n_estimators': hp.qlognormal('n_estimators', 4, 1, 1),
            'max_depth': hp.choice('use_max_depth',
                                   [None, hp.qlognormal('max_depth', 3, 1, 1)]),
            'min_samples_leaf': hp.choice('use_min_samples_leaf',
                                [1 , hp.qlognormal('min_samples_split', 2, 1, 1) + 1])
        })

    def train(self, x, y,
              n_estimators=10,
              max_depth=None,
              min_samples_leaf=None):
        n_estimators = self.to_int(n_estimators)
        max_depth = self.to_int(max_depth)
        min_samples_leaf = self.to_int(min_samples_leaf)

        self.rf_model = RandomForestClassifier(n_estimators,
                                               max_depth=max_depth,
                                               min_samples_leaf=min_samples_leaf)
        self.rf_model.fit(x, y)

    def predict(self, x):
        return self.rf_model.predict_proba(x)[:, 1]

