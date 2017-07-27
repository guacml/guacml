from guacml.metrics.accuracy import Accuracy
from guacml.metrics.mean_absolute_error import MeanAbsoluteError
from guacml.metrics.root_mean_squared_log_error import RootMeanSquaredLogError
from guacml.metrics.root_mean_squared_percentage_error import RootMeanSquaredPercentageError
from guacml.metrics.log_loss import LogLoss
from guacml.metrics.mean_squared_error import MeanSquaredError


def eval_metric_from_name(metric_name):
    if metric_name == 'accuracy':
        return Accuracy()
    elif metric_name == 'logloss':
        return LogLoss()
    elif metric_name == 'mse' or metric_name == 'mean_square_error':
        return MeanSquaredError()
    elif metric_name == 'rmsle' or metric_name == 'root_mean_squared_log_error':
        return RootMeanSquaredLogError()
    elif metric_name == 'mae' or metric_name == 'mean_absolute_error':
        return MeanAbsoluteError()
    elif metric_name == 'rmspe' or metric_name == 'root_mean_square_percentage_error':
        return RootMeanSquaredPercentageError()
    else:
        raise NotImplementedError('Unknown eval metric: ' + metric_name)
