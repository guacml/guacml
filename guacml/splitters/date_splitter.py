import pandas as pd


class DateSplitter:
    def __init__(self, config):
        self.config = config
        self.n_folds = config['cross_validation']['n_folds']
        ts_conf = config['run_time']['time_series']
        self.date_split_col = ts_conf['date_split_col']
        self.prediction_length = ts_conf['prediction_length']
        self.frequency = ts_conf['frequency']
        self.n_offset_models = ts_conf['n_offset_models']


    def holdout_split(self, input):
        """
        In the case of several models to predict different offsets in the future,
        we do cross validation on the length of a single model. Only for the holdout
        set we use the full length and predict with many models.
        """
        dates = input[self.date_split_col]
        split_point = dates.max() - (self.frequency * self.prediction_length * self.n_offset_models)
        return input[dates < split_point], input[dates >= split_point]

    def cv_splits(self, input):
        dates = input[self.date_split_col]
        left = dates.max()
        split_points = []
        for i in range(self.n_folds):
            right = left
            left = left - pd.Timedelta(days=self.prediction_length)
            split_points.append((left, right))
        split_points.reverse()
        if split_points[0][0] - dates.min() < pd.Timedelta(days=self.prediction_length):
            raise Exception('Training set is shorter than the prediction length. Use a less'
                            'cross validation folds or a shorter prediction length')

        split_indices = []
        for left, right in split_points:
            train = input[dates < left]
            cv = input[(dates >= left) & (dates < right)]
            split_indices.append([train.index.values, cv.index.values])

        return split_indices
