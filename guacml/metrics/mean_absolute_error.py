from guacml.metrics.base_eval_metric import BaseEvalMetric


class MeanAbsoluteError(BaseEvalMetric):

    @staticmethod
    def error(truth, prediction):
        return ((truth - prediction).abs()).mean()

    @staticmethod
    def row_wise_error(truth, prediction):
        return (truth - prediction).abs()
