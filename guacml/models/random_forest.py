from guacml.models.base_model import BaseModel
from guacml.models.hyper_param_info import HyperParameterInfo
from guacml.preprocessing.column_analyzer import ColType
from sklearn.ensemble import RandomForestClassifier

N_ESTIMATORS_DEFAULT = 50
MIN_SAMPLES_LEAF_DEFAULT = 1


class RandomForest(BaseModel):
    def get_valid_types(self):
        return [ColType.BINARY, ColType.NUMERIC, ColType.ORDINAL, ColType.INT_ENCODING]

    @staticmethod
    def hyper_parameter_info():
        return {
            'n_estimators': HyperParameterInfo(N_ESTIMATORS_DEFAULT,
                                               [10, 1000],
                                               [10, 50, 100]),
            'min_samples_leaf': HyperParameterInfo(MIN_SAMPLES_LEAF_DEFAULT,
                                                   [1, 1000],
                                                   [1, 10, 50])
        }

    def train(self, x, y, n_estimators=N_ESTIMATORS_DEFAULT, min_samples_leaf=MIN_SAMPLES_LEAF_DEFAULT):
        self.rf_model = RandomForestClassifier(self.pos_int(n_estimators),
                                               min_samples_leaf=self.pos_int(min_samples_leaf))
        self.rf_model.fit(x, y)

    def predict(self, x):
        return self.rf_model.predict_proba(x)

    @staticmethod
    def pos_int(value):
        return max(int(value), 0)