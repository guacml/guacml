class BaseEvalMetric:
    @staticmethod
    def error(truth, prediction):
        raise NotImplementedError()

    @staticmethod
    def row_wise_error(truth, prediction):
        raise NotImplementedError()

