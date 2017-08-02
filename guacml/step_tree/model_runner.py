import pandas as pd
import numpy as np

from hyperopt import STATUS_OK
from guacml import splitters
from guacml import metrics
from guacml.step_tree.lagged_target_handler import LaggedTargetHandler


class ModelRunner():
    def __init__(self, model, data, config, logger):
        self.model = model
        self.logger = logger
        self.metadata = data.metadata
        rt_conf = config['run_time']
        self.target = rt_conf['target']
        self.eval_metric = metrics.create(rt_conf['eval_metric'])
        if rt_conf['target_transform'] is not None:
            self.target_trans = rt_conf['target_transform']
            data.df['guac_transformed_target'] = self.target_trans.transform(data.df[self.target])

        self.prediction_range = rt_conf['prediction_range']
        if rt_conf['is_time_series']:
            self.ts_config = rt_conf['time_series']
            self.n_offset_models = self.ts_config['n_offset_models']

        splitter = splitters.create(config)

        holdout_train, holdout = splitter.holdout_split(data.df)

        self.holdout = holdout.copy()
        self.train_and_cv = holdout_train.copy()
        self.train_and_cv_folds = list(splitter.cv_splits(self.train_and_cv))

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
        self.logger.info('Training %s on %d folds', self.model.name(), len(self.train_and_cv_folds))
        fold_number = 1

        for train_indices, cv_indices in self.train_and_cv_folds:
            x = self.train_and_cv[features].loc[train_indices]
            if not hasattr(self, 'target_trans'):
                y = self.train_and_cv[self.target].loc[train_indices]
            else:
                y = self.train_and_cv['guac_transformed_target'].loc[train_indices]
            self.logger.info('Training %s fold #%d on %d rows',
                             self.model.name(), fold_number, x.shape[0])
            self.model.train(x, y, **hyper_params)

            prediction = self.model.predict(self.train_and_cv[features].loc[cv_indices])
            if prediction.isnull().any():
                raise Exception('Some predictions where N/A')

            if not hasattr(self, 'target_trans'):
                self.train_and_cv.loc[cv_indices, 'cv_prediction'] = prediction
            else:
                self.train_and_cv.loc[cv_indices, 'cv_transformed_prediction'] = prediction
                transformed_back = self.target_trans.transform_back(prediction)
                self.train_and_cv.loc[cv_indices, 'cv_prediction'] = transformed_back

            if with_feature_importances:
                feat_importance = self.model.feature_importances(self.train_and_cv[features])
                if feat_importance is None or len(feat_importance) == 0:
                    raise Exception('Error computing feature importances.')
                feature_importances.append(feat_importance)

            fold_number += 1

        self._truncate_predictions(self.train_and_cv, 'cv_prediction')

        if with_feature_importances:
            feature_importances = pd.DataFrame(feature_importances)
            self.cv_feature_importances = feature_importances.mean()

    def train_and_predict_with_holdout_model(self, features, hyper_params):
        self.logger.info('Training holdout model %s on features %s using %s',
                         self.model.name(), list(features), hyper_params)

        x = self.train_and_cv[features]
        if not hasattr(self, 'target_trans'):
            y = self.train_and_cv[self.target]
        else:
            y = self.train_and_cv['guac_transformed_target']

        self.model.train(x, y, **hyper_params)

        prediction = self.model.predict(self.holdout[features])
        if prediction.isnull().any():
            raise Exception('Some predictions where N/A')

        if not hasattr(self, 'target_trans'):
            self.holdout['prediction'] = prediction
        else:
            self.holdout['transformed_prediction'] = prediction
            self.holdout['prediction'] = self.target_trans.transform_back(prediction)
        self._truncate_predictions(self.holdout, 'prediction')

    def train_and_predict_with_offset_models(self, features, hyper_params):
        for i_offset in range(self.n_offset_models):
            train = self.train_and_cv
            train, features = LaggedTargetHandler.select_offset_features(train,
                                                                         self.metadata,
                                                                         features,
                                                                         offset=0)
            if not hasattr(self, 'target_trans'):
                y = self.train_and_cv[self.target]
            else:
                y = self.train_and_cv['guac_transformed_target']
            self.model.train(train[features], y, **hyper_params)

            offset_labels = LaggedTargetHandler.holdout_offset_labels(self.ts_config,
                                                                      self.holdout,
                                                                      i_offset)

            prediction = self.model.predict(self.holdout.loc[offset_labels, features])
            if prediction.isnull().any():
                raise Exception('Some predictions where N/A')
            if not hasattr(self, 'target_trans'):
                self.holdout.loc[offset_labels, 'prediction'] = prediction
            else:
                self.holdout.loc[offset_labels, 'transformed_prediction'] = prediction
                self.holdout.loc[offset_labels, 'prediction'] = \
                    self.target_trans.transform_back(prediction)

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
