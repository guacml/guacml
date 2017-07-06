class FeatureReducer():
    def __init__(self, model_runner, hyper_params, logger):
        self.model_runner = model_runner
        self.hyper_params = hyper_params
        self.logger = logger

    def reduce(self, features):
        if len(features) == 1:
            return features

        self.model_runner.train_for_cv(features, self.hyper_params, True)
        cv_bs_errors = self.model_runner.bootstrap_cv_errors()

        features = self.reduce_using_feature_importance(features, cv_bs_errors)
        features = self.reduce_one_by_one(features, cv_bs_errors)
        return features

    def reduce_using_feature_importance(self, features, cv_bs_errors):
        max_iterations = 10
        for i in range(max_iterations):
            feat_imp = self.model_runner.cv_feature_importances
            feat_imp = feat_imp[features].sort_values()
            cumsums = feat_imp.cumsum() / feat_imp.sum()
            reduced_feats = cumsums[cumsums <= 0.95].index

            self.model_runner.train_for_cv(reduced_feats, self.hyper_params)
            if self.model_runner.is_cv_error_significantly_worse(cv_bs_errors):
                break
            else:
                self.logger.info('Dropped features %s', set(features) - set(reduced_feats))
                features = reduced_feats
                if len(reduced_feats) == 1:
                    break
        return features

    def reduce_one_by_one(self, features, cv_bs_errors):
        while len(features) > 1:
            for to_drop in features:
                reduced_feats = list(features)
                reduced_feats.remove(to_drop)
                self.model_runner.train_for_cv(reduced_feats, self.hyper_params)
                if not self.model_runner.is_cv_error_significantly_worse(cv_bs_errors):
                    self.logger.info('Dropped feature %s', to_drop)
                    features = reduced_feats
                    break

            return features
        return features
