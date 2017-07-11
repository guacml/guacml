from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold


class RandomSplitter:
    def __init__(self, cv_config):
        self.cv_config = cv_config

    def holdout_split(self, input):
        return train_test_split(input, test_size=self.cv_config['holdout_size'])

    def cv_splits(self, input):
        n_folds = self.cv_config['n_folds']

        if n_folds > 1:
            k_fold = KFold(n_splits=n_folds)
            splits = []

            for train_indices, cv_indices in k_fold.split(input):
                splits.append([
                    self.positional_to_label_index(train_indices, input),
                    self.positional_to_label_index(cv_indices, input)
                ])

            return splits
        else:
            train, cv = train_test_split(input, test_size=self.cv_config['cv_size'])
            return [[train.index.values, cv.index.values]]

    @staticmethod
    def positional_to_label_index(indices, df):
        return df.iloc[indices].index.values
