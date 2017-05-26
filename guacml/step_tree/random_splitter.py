from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold


class RandomSplitter:
    def __init__(self, ratio, n_folds=3):
        self.ratio = ratio
        self.k_fold = KFold(n_splits=n_folds)

    def holdout_split(self, input):
        return train_test_split(input, train_size=self.ratio)

    def cv_splits(self, input):
        return self.k_fold.split(input)
