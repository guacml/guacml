# TODO implement linear, base model class
class LinearModel:
    def select_features(self, metadata):
        return metadata.col_name

    def get_adapter(self):
        return Adapter()

class Adapter:
    def train(self, x, y):
        pass

    def predict(self, x):
        return [0] * (x.shape[0] - 1) + [1]
