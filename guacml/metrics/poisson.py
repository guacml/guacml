from guacml.metrics.base_eval_metric import BaseEvalMetric
import numpy as np
import math


class Poisson(BaseEvalMetric):

    @staticmethod
    def error(truth, prediction):
        return Poisson.row_wise_error(truth, prediction).mean()

    @staticmethod
    def row_wise_error(truth, prediction):
        eps = 1e-15
        prediction = np.clip(prediction, eps, None)
        lgamma = list(map(math.lgamma, truth + 1))
        print(lgamma)
        print(prediction)
        print(np.log(prediction))
        print(truth)
        return lgamma + prediction - np.log(prediction) * truth
