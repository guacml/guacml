from guacml.metrics.base_eval_metric import BaseEvalMetric
import numpy as np
from sklearn.metrics import log_loss


class LogLoss(BaseEvalMetric):

    @staticmethod
    def error(truth, prediction):
        return LogLoss.row_wise_error(truth, prediction).mean()

    @staticmethod
    def row_wise_error(truth, prediction):
        eps = 1e-15
        prediction = np.clip(prediction, eps, 1 - eps)
        return -1 * (truth * np.log(prediction) + (1 - truth) * np.log(1 - prediction))

