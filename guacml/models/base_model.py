class BaseModel:
    def __init__(self, problem_type):
        self.problem_type = problem_type

    def get_valid_types(self):
        raise NotImplementedError()

    @staticmethod
    def hyper_parameter_info():
        raise NotImplementedError()

    def train(self, x, y, **kwargs):
        raise NotImplementedError()

    def predict(self, x):
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
        return max(int(value), 0)
