from guacml.metrics.base_eval_metric import BaseEvalMetric

THRESHOLD = 0.5


class Accuracy(BaseEvalMetric):
    @staticmethod
    def row_wise_error(truth, prediction):
        return -Accuracy.correct(truth, prediction)

    @staticmethod
    def error(truth, prediction):
        return -Accuracy.correct(truth, prediction).sum() / truth.shape[0]

    @staticmethod
    def correct(truth, prediction):
        return (prediction >= THRESHOLD) == (truth == 1)