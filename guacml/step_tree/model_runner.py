import pandas as pd
import numpy as np

from hyperopt import STATUS_OK
from guacml import splitters


class ModelRunner():
    def __init__(self, model, data, config):
        self.model = model

        self.target = config['run_time']['target']
        self.eval_metric = config['run_time']['eval_metric']
        self.prediction_range = config['run_time']['prediction_range']
        splitter = splitters.create(config)

        holdout_train, holdout = splitter.holdout_split(data.df)

        self.holdout = holdout.copy()
        self.train_and_cv = holdout_train.copy()
        self.train_and_cv_folds = list(splitter.cv_splits(self.train_and_cv))
        self.final_features = None
        self.final_hyper_params = None

    def train_and_cv_error(self, features, hyper_params):
        self.train_for_cv(features, hyper_params)
        target = self.train_and_cv[self.target]
        prediction = self.train_and_cv['cv_prediction']
        if prediction.isnull().any():
            Exception('Some predictions where N/A.')
        self._truncate_predictions(self.train_and_cv, 'cv_prediction')

        loss = self.eval_metric.error(target, prediction)
        loss_variance = self.bootstrap_errors_(target, prediction).var()
        if loss is None or np.isnan(loss) or loss_variance is None or np.isnan(loss_variance):
            raise Exception('Could not calculate cv error.')
        return {
            'status': STATUS_OK,
            'loss': loss,
            'loss_variance': loss_variance
        }

    def _truncate_predictions(self, data, prediction_col):
        if self.prediction_range is not None:
            pred_min, pred_max = self.prediction_range
            if pred_min is not None:
                data.loc[data[prediction_col] < pred_min, prediction_col] = pred_min
            if pred_max is not None:
                data.loc[data[prediction_col] > pred_max, prediction_col] = pred_max

    def train_for_cv(self, features, hyper_params, with_feature_importances=False):
        feature_importances = []
        self.train_and_cv['cv_prediction'] = np.nan
        for train_indices, cv_indices in self.train_and_cv_folds:

            self.model.train(self.train_and_cv[features].loc[train_indices],
                             self.train_and_cv[self.target].loc[train_indices],
                             **hyper_params)

            prediction = self.model.predict(self.train_and_cv[features].loc[cv_indices])
            if prediction.isnull().any():
                raise Exception('Some predictions where N/A')
            self.train_and_cv.loc[cv_indices, 'cv_prediction'] = prediction

            if with_feature_importances:
                feat_importance = self.model.feature_importances(self.train_and_cv[features])
                if feat_importance is None or len(feat_importance) == 0:
                    raise Exception('Error computing feature importances.')
                feature_importances.append(feat_importance)

        self._truncate_predictions(self.train_and_cv, 'cv_prediction')

        if with_feature_importances:
            feature_importances = pd.DataFrame(feature_importances)
            self.cv_feature_importances = feature_importances.mean()

    def train_final_model(self, features, hyper_params):
        self.final_features = features
        self.final_hyper_params = hyper_params
        self.model.train(
                self.train_and_cv[features],
                self.train_and_cv[self.target],
                **hyper_params
            )

        train_prediction = self.model.predict(self.train_and_cv[features])
        if train_prediction.isnull().any():
            raise Exception('Some predictions where N/A')
        self.train_and_cv['train_prediction'] = train_prediction
        self._truncate_predictions(self.train_and_cv, 'train_prediction')
        holdout_prediction = self.model.predict(self.holdout[features])
        if holdout_prediction.isnull().any():
            raise Exception('Some predictions where N/A')
        self.holdout['prediction'] = holdout_prediction
        self._truncate_predictions(self.holdout, 'prediction')

    def cv_error(self):
        return self.eval_metric.error(self.train_and_cv[self.target],
                                      self.train_and_cv.cv_prediction)

    def training_error(self):
        return self.eval_metric.error(self.train_and_cv[self.target],
                                      self.train_and_cv.cv_prediction)

    def holdout_error(self):
        return self.eval_metric.error(self.holdout[self.target], self.holdout.prediction)

    def cv_predictions(self):
        return self.train_and_cv.cv_prediction

    def holdout_predictions(self):
        return self.holdout.prediction

    def row_wise_holdout_error(self):
        return self.eval_metric.row_wise_error(self.holdout[self.target],
                                               self.holdout_predictions())

    def holdout_error_interval(self):
        bs_holdout_errors = self.bootstrap_errors_(self.holdout[self.target],
                                                   self.holdout_predictions())

        return bs_holdout_errors.quantile(0.1), bs_holdout_errors.quantile(0.9)

    def bootstrap_cv_errors(self):
        return self.bootstrap_errors_(self.train_and_cv[self.target], self.cv_predictions())

    def bootstrap_errors_(self, truth, predictions, i=200):
        n = len(truth)
        errors = []

        for i in range(i):
            resample_indices = np.random.randint(n, size=n)
            err = self.eval_metric.error(truth.iloc[resample_indices],
                                         predictions.iloc[resample_indices])
            errors.append(err)

        return pd.Series(errors)

    def is_cv_error_significantly_worse(self, other_errors):
        new_bs_errors = self.bootstrap_errors_(self.train_and_cv[self.target],
                                               self.cv_predictions())
        # error gets larger with probability larger 80%
        return (new_bs_errors > other_errors).mean() > 0.8

    def hyper_parameter_info(self):
        return self.model.hyper_parameter_info()
