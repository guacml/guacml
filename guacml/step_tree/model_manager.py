from guacml.step_tree.feature_reducer import FeatureReducer
from guacml.step_tree.hyper_param_optimizer import HyperParameterOptimizer
from guacml.step_tree.model_result import ModelResult
from guacml.step_tree.model_runner import ModelRunner
from guacml.step_tree.lagged_target_handler import LaggedTargetHandler


class ModelManager():
    def __init__(self, model, config, logger):
        self.model = model
        self.config = config
        self.target = config['run_time']['target']
        self.logger = logger
        self.lagged_target_features = None

    def execute(self, data):
        features = self.select_features(data.metadata)
        features = features[features != self.target]
        if self.config['run_time']['is_time_series'] and \
           self.config['run_time']['time_series']['n_offset_models'] > 1:
            data.df, features = LaggedTargetHandler.select_offset_features(data.df,
                                                                           data.metadata,
                                                                           features,
                                                                           offset=0)

        model_runner = ModelRunner(self.model, data, self.config, self.logger)
        hp_optimizer = HyperParameterOptimizer(model_runner, features)
        all_trials, best_hps = hp_optimizer.optimize(
            self.config['run_time']['hyper_param_iterations']
        )

        if self.config['model_manager']['reduce_features'] is True:
            feature_reducer = FeatureReducer(model_runner, best_hps, self.logger)
            features = feature_reducer.reduce(features)

        return self.build_result(model_runner, features, all_trials, best_hps)

    def select_features(self, metadata):
        return metadata[metadata.type.isin(self.model.get_valid_types())].index.values

    def build_result(self, model_runner, features, all_trials, best_hps):
        df_trials = HyperParameterOptimizer.trials_to_data_frame(all_trials)
        df_trials = df_trials.sort_values('cv error')
        best = df_trials.iloc[0]

        if not self.config['run_time']['is_time_series'] or \
                self.config['run_time']['time_series']['n_offset_models'] == 1:
            model_runner.train_and_predict_with_holdout_model(features, best_hps)
            training_error = model_runner.training_error()
        else:
            model_runner.train_and_predict_with_offset_models(features, best_hps)
            training_error = None

        holdout_predictions = model_runner.holdout_predictions()
        holdout_error = model_runner.holdout_error()
        holdout_error_interval = model_runner.holdout_error_interval()
        holdout_row_errors = model_runner.row_wise_holdout_error()

        holdout = model_runner.holdout.copy()
        holdout['prediction'] = holdout_predictions
        holdout['error'] = holdout_row_errors
        metadata = model_runner.metadata

        return ModelResult(self.model,
                           features,
                           self.target,
                           training_error,
                           best['cv error'],
                           holdout_error,
                           holdout_error_interval,
                           holdout,
                           metadata,
                           best,
                           df_trials)
