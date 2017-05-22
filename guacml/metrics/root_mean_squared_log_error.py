import math
import numpy as np
from guacml.metrics.base_eval_metric import BaseEvalMetric


def _row_error(truth, prediction):
    return (np.log1p(prediction) - np.log1p(truth)) ** 2


class RootMeanSquaredLogError(BaseEvalMetric):

    @staticmethod
    def error(truth, prediction):
        return math.sqrt(_row_error(truth, prediction).mean())

    @staticmethod
    def row_wise_error(truth, prediction):
        return np.sqrt(_row_error(truth, prediction))
