class LaggedTargetHandler:

    def select_offset_features(df, meta, features, offset):
        """
        When predicting time series many steps into the future, we are using several models for
        different numbers of steps in the future (offsets), because a model that predicts a month in the
        future can only use aggregations of the target that are at least a month old.

        We only do hyper parameter optimization and feature selection once. For the feature
        selection we map the selected aggregations for different offsets to the same column name.

        This method selects the appropriate lagged aggregations of the target variable for the
        specified model offset. It mutates the input DataFrame to store the selected column under
        the shared name.
        """
        feat_meta = meta[meta.index.isin(features)]
        non_lagged_features = feat_meta[feat_meta['is_lagged_target'].isnull()]
        lagged_features = feat_meta[feat_meta['lagged_target_model_offset'] == offset]

        for idx, lagged_feat_row in lagged_features.iterrows():
            df[lagged_feat_row['lagged_target_shared_name']] = df[lagged_feat_row.name]

        features = non_lagged_features.index.tolist() + lagged_features['lagged_target_shared_name'].tolist()
        return df, features

    @staticmethod
    def holdout_offset_labels(time_series_config, holdout, offset):
        date_split_col = time_series_config['date_split_col']
        prediction_length = time_series_config['prediction_length']
        frequency = time_series_config['frequency']

        dates = holdout[date_split_col]
        lower = dates.min() + (frequency * (prediction_length * offset))
        upper = dates.min() + (frequency * (prediction_length * (offset + 1) - 1))
        return holdout[dates.between(lower, upper)].index



