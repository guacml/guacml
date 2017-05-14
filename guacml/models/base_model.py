class BaseModel:
    def select_features(self, metadata):
        return metadata[metadata.type.isin(self.get_valid_types())].col_name

    def get_valid_types(self):
        raise NotImplementedError()

    def get_adapter(self):
        raise NotImplementedError()


class BaseAdapter:
    def train(self, x, y):
        raise NotImplementedError()

    def predict(self, x):
        raise NotImplementedError()