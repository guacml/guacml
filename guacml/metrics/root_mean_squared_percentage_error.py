import math
import numpy as np
from guacml.metrics.base_eval_metric import BaseEvalMetric


def row_error(truth, prediction):
    if (truth == 0).any():
        raise Exception('Can not compute rmspe, when there are zeros in the target column.')
    return ((truth - prediction) / truth) ** 2


class RootMeanSquaredPercentageError(BaseEvalMetric):

    @staticmethod
    def error(truth, prediction):
        return math.sqrt(row_error(truth, prediction).mean())

    @staticmethod
    def row_wise_error(truth, prediction):
        return np.sqrt(row_error(truth, prediction))
