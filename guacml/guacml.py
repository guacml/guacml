import os
import yaml
import pandas as pd

from guacml.dataset import Dataset
from guacml.enums import ProblemType, ColType
from guacml.metrics.accuracy import Accuracy
from guacml.metrics.log_loss import LogLoss
from guacml.metrics.mean_absolute_error import MeanAbsoluteError
from guacml.metrics.mean_squared_error import MeanSquaredError
from guacml.metrics.root_mean_squared_log_error import RootMeanSquaredLogError
from guacml.metrics.root_mean_squared_percentage_error import RootMeanSquaredPercentageError
from guacml.plots import Plots
from guacml.step_tree.tree_builder import TreeBuilder
from guacml.step_tree.tree_runner import TreeRunner


class GuacMl:
    def __init__(self, data, target, eval_metric=None, exclude_cols=None, config=None):
        conf_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
        with open(conf_path, 'r') as file:
            self.config = yaml.load(file)

        if config is not None:
            self.config.update(config) # TODO: don't replace whole nested dicts

        self.data = Dataset.from_df(data, target, exclude_cols)

        metadata = self.data.metadata
        target_meta = metadata.loc[target]
        rt_conf = self.config['run_time']
        if target_meta.type == ColType.BINARY:
            problem_type = ProblemType.BINARY_CLAS
            rt_conf['eval_metric'] = LogLoss()
            print('Binary classification problem detected.')
        elif target_meta.type in [ColType.CATEGORICAL, ColType.INT_ENCODING]:
            problem_type = ProblemType.MULTI_CLAS
            rt_conf['eval_metric'] = LogLoss()
            print('Multi class classification problem detected.')
        elif target_meta.type in [ColType.ORDINAL, ColType.NUMERIC]:
            problem_type = ProblemType.REGRESSION
            rt_conf['eval_metric'] = MeanSquaredError()
            print('Regression problem detected.')
        else:
            raise Exception('Can not automatically infer problem type.')

        if eval_metric is not None:
            metric_name = eval_metric.lower()

            if metric_name == 'accuracy':
                rt_conf['eval_metric'] = Accuracy()
            elif metric_name == 'rmsle' or metric_name == 'root_mean_squared_log_error':
                rt_conf['eval_metric'] = RootMeanSquaredLogError()
            elif metric_name == 'mae' or metric_name == 'mean_absolute_error':
                rt_conf['eval_metric'] = MeanAbsoluteError()
            elif metric_name == 'rmspe' or metric_name == 'root_mean_square_percentage_error':
                rt_conf['eval_metric'] = RootMeanSquaredPercentageError()
            else:
                raise NotImplementedError('Unknown eval metric: ' + eval_metric)

        rt_conf['problem_type'] = problem_type
        rt_conf['target'] = target
        rt_conf['exclude_cols'] = exclude_cols

        tree_builder = TreeBuilder(self.config)
        self.tree = tree_builder.build()
        self.plots = Plots(rt_conf, self.data, self.tree)
        self.model_results = None
        self.runner = None

    def make_time_series(self, date_split_col, series_key_cols=None, prediction_length=1):
        if date_split_col not in self.data.df.columns:
            raise Exception('The date_split_col {} was not in the columns of the data set {}.'
                            .format(date_split_col, self.data.df.columns))
        if series_key_cols is None:
            series_key_cols = ['guac_time_series_key']
            # this hack makes the whole code that groups by time series key reusable for the case of a single time series.
            # it's omitted in the metadata
            self.data.df['guac_time_series_key'] = True
        elif not isinstance(series_key_cols, list):
            series_key_cols = [series_key_cols]
        for key_col in series_key_cols:
            if key_col not in self.data.df.columns:
                raise Exception('The time series key column {} was not in the columns of the data set {}.'
                                .format(key_col, self.data.df.columns))
        if not isinstance(prediction_length, int) and prediction_length > 0:
            raise Exception('Prediction length must be positive integer, but was {}'.format(prediction_length))

        rt_conf = self.config['run_time']
        rt_conf['is_time_series'] = True
        rt_conf['time_series']['date_split_col'] = date_split_col
        rt_conf['time_series']['series_key_cols'] = series_key_cols
        rt_conf['time_series']['prediction_length'] = prediction_length

        # ToDo: this is duplicated from the guac constructor
        tree_builder = TreeBuilder(self.config)
        self.tree = tree_builder.build()
        self.plots = Plots(rt_conf, self.data, self.tree)

    def run(self, hyper_param_iterations, prediction_range=None):
        if not isinstance(hyper_param_iterations, int) and hyper_param_iterations > 0:
            raise Exception('Number of hyper parameter iterations must be positive integer,'
                            ' but was {}'.format(hyper_param_iterations))

        # TODO: we shouldn't be mutating config
        self.config['run_time']['hyper_param_iterations'] = hyper_param_iterations
        self.config['run_time']['prediction_range'] = prediction_range

        self.runner = TreeRunner(self.data, self.config, self.tree)
        self.model_results = self.runner.run()

        self.plots.set_model_results(self.model_results)

        return self.model_overview()

    def model_overview(self):
        rows = []
        for name, res in self.model_results.items():
            res_dict = res.to_display_dict()
            res_dict['model name'] = name
            rows.append(res_dict)

        columns = ['model name', 'n features', 'holdout error', 'holdout error interval',
                   'cv error', 'training error']
        result = pd.DataFrame(rows, columns=columns + ['holdout error numeric'])
        return result.sort_values('holdout error numeric')[columns]

    def info(self):
        return self.data.display_metadata()

    def clear_previous_runs(self):
        if self.runner is not None:
            self.runner.clear_prev_runs()

    def hyper_param_runs(self, model_name):
        if model_name in self.model_results:
            return self.model_results[model_name].all_hyper_param_runs
        else:
            raise ValueError('Model name has to be in {0}'.format(self.model_results.keys()))
