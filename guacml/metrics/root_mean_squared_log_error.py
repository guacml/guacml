import math
import numpy as np
from guacml.metrics.base_eval_metric import BaseEvalMetric


def _row_error(truth, prediction):
    if (truth == 0).any():
        raise ValueError('Can not compute rmsle, '
                         'when there are zeros in the target column.')
    if (truth < 0).any():
        raise ValueError('Can not compute rmsle, '
                         'when there are negative values in the target column.')
    if (prediction < 0).any():
        raise ValueError('Can not compute rmsle, '
                         'when there are negative values in the predictions.')

    return (np.log(np.maximum(prediction, 0)) - np.log(truth)) ** 2


class RootMeanSquaredLogError(BaseEvalMetric):

    @staticmethod
    def error(truth, prediction):
        return math.sqrt(_row_error(truth, prediction).mean())

    @staticmethod
    def row_wise_error(truth, prediction):
        return np.sqrt(_row_error(truth, prediction))
