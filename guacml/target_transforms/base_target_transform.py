class BaseTargetTransform():
    def transform(self, target_col):
        raise NotImplementedError()

    def transform_back(self, transformed_target):
        raise NotImplementedError()

