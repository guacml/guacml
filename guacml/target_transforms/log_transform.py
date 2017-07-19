from guacml.target_transforms.base_target_transform import BaseTargetTransform
import numpy as np


class LogTransform(BaseTargetTransform):
    def transform(self, target_col):
        return np.log1p(target_col)

    def transform_back(self, transformed_target):
        return np.expm1(transformed_target)
