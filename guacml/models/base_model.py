class BaseModel:

    def __init__(self, config, logger):
        self.config = config
        self.problem_type = config['run_time']['problem_type']
        self.logger = logger

    def copy(self, config):
        copy = self.__class__(config, self.logger)
        copy.model = self.copy_model()

        return copy

    def copy_model(self):
        raise NotImplementedError()

    def name(self):
        return self.__class__.__name__

    def get_valid_types(self):
        raise NotImplementedError()

    def hyper_parameter_info(self):
        raise NotImplementedError()

    def train(self, x, y, **kwargs):
        raise NotImplementedError()

    def predict(self, x):
        """
        :return: Prediction as a pandas Series with the index of x.
        """
        raise NotImplementedError()

    def feature_importances(self, x):
        raise NotImplementedError()

    @staticmethod
    def to_int(value):
        if value is None:
            return None
        return int(value)

    @staticmethod
    def pos_int(value):
        return max(int(value), 1)
