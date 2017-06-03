from sklearn.model_selection import train_test_split


class RandomSplitter:
    def __init__(self, ratio):
        self.ratio = ratio

    def split(self, input):
        return train_test_split(input, train_size=self.ratio)
