from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold


class RandomSplitter:
    def __init__(self, cv_config):
        self.test_size = cv_config['holdout_size']
        self.k_fold = KFold(n_splits=cv_config['n_folds'])

    def holdout_split(self, input):
        return train_test_split(input, test_size=self.test_size)

    def cv_splits(self, input):
        for train_indices, cv_indices in self.k_fold.split(input):
            yield self.positional_to_label_index(train_indices, input), \
                  self.positional_to_label_index(cv_indices, input)

    @staticmethod
    def positional_to_label_index(indices, df):
        return df.iloc[indices].index.values
