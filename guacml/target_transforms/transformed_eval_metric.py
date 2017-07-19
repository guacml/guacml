from guacml.metrics.base_eval_metric import BaseEvalMetric


class TransformedEvalMetric(BaseEvalMetric):
    def __init__(self, eval_metric, target_transform):
        self.inner_metric = eval_metric
        self.target_transform = target_transform

    def error(self, truth, prediction):
        back_trans_truth = self.target_transform.transform_back(truth)
        back_trans_prediction = self.target_transform.transform_back(prediction)
        return self.inner_metric.error(back_trans_truth, back_trans_prediction)

    def row_wise_error(self, truth, prediction):
        back_trans_truth = self.target_transform.transform_back(truth)
        back_trans_prediction = self.target_transform.transform_back(prediction)
        return self.inner_metric.row_wise_error(back_trans_truth, back_trans_prediction)
