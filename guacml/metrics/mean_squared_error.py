from guacml.metrics.base_eval_metric import BaseEvalMetric


class MeanSquaredError(BaseEvalMetric):

    @staticmethod
    def error(truth, prediction):
        return ((truth - prediction) ** 2).mean()

    @staticmethod
    def row_wise_error(truth, prediction):
        return (truth - prediction) ** 2
