import numpy as np

class DateSplitter:
    def __init__(self, config):
        self.config = config

    def holdout_split(self, input):
        series = input[self.config['run-time']['date_split_col']]
        split_point = series.quantile(1 - self.config['cross-validation']['holdout_size'], interpolation='lower')
        return input[series <= split_point], input[series > split_point]

    def cv_splits(self, input):
        n_folds = self.config['n_folds']
        boundaries = np.delete(np.linspace(0, 1, n_folds + 1), 0)
        series = input[self.config['date_split_col']]
        splits = []

        for i in range(len(boundaries) - 1):
            left = boundaries[i]
            right = boundaries[i + 1]
            left_split = series.quantile(left)
            right_split = series.quantile(right, interpolation='higher')
            train = input[series <= left_split]
            cv = input[(series > left_split) & (series <= right_split)]
            splits.append([train.index.values, cv.index.values])

        return splits
