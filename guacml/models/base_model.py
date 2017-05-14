class BaseModel:
    def select_features(self, metadata):
        return metadata[metadata.type.isin(self.get_valid_types())].col_name

    def get_valid_types(self):
        raise NotImplementedError()

    @staticmethod
    def hyper_parameter_info():
        raise NotImplementedError()

    def train(self, x, y, **kwargs):
        raise NotImplementedError()

    def predict(self, x):
        raise NotImplementedError()