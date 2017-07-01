import pandas as pd


class DateSplitter:
    def __init__(self, config):
        self.config = config
        rt_conf = self.config['run_time']
        self.date_split_col = rt_conf['time_series']['date_split_col']
        self.prediction_length = rt_conf['time_series']['prediction_length']
        self.n_folds = self.config['cross_validation']['n_folds']

    def holdout_split(self, input):
        dates = input[self.date_split_col]
        split_point = dates.max() - pd.Timedelta(days=self.prediction_length)
        return input[dates < split_point], input[dates >= split_point]

    def cv_splits(self, input):
        dates = input[self.date_split_col]
        left = dates.max()
        split_points = []
        for i in range(self.prediction_length):
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
