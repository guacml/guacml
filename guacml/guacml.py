import os
import yaml
import pandas as pd
import logging

from guacml.dataset import Dataset
import guacml.target_transforms as transforms

from guacml.plots import Plots
from guacml.step_tree.tree_builder import TreeBuilder
from guacml.step_tree.tree_runner import TreeRunner
from guacml.util import deep_update
from guacml.util.time_series_util import analyze_frequency
from guacml.pipeline import Pipeline


class GuacMl:
    def __init__(self, data, target, eval_metric=None, exclude_cols=None, config=None,
                 problem_type=None, target_transform=None, log_level=None):
        conf_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
        with open(conf_path, 'r') as file:
            self.config = yaml.load(file)

        if config is not None:
            deep_update(self.config, config)

        self.logger = logging.getLogger(__name__)
        if log_level is None or log_level == 'info':
            self.logger.setLevel(logging.INFO)
        elif log_level == 'warning':
            self.logger.setLevel(logging.WARNING)
        elif log_level == 'error':
            self.logger.setLevel(logging.ERROR)
        else:
            raise Exception('Log level {} not knonw.'.format(log_level))

        self.data = Dataset.from_df(data, self.config, target, exclude_cols, self.logger)

        metadata = self.data.metadata
        target_meta = metadata.loc[target]
        rt_conf = self.config['run_time']

        if problem_type is None:
            if target_meta.type == 'binary':
                problem_type = 'binary_clas'
                self.logger.info('Binary classification problem detected.')
            elif target_meta.type in ['categorical', 'int_encoding']:
                problem_type = 'multi_clas'
                self.logger.info('Multi class classification problem detected.')
            elif target_meta.type in ['ordinal', 'numeric']:
                problem_type = 'regression'
                self.logger.info('Regression problem detected.')
            else:
                raise Exception('Can not automatically infer problem type.')

        if eval_metric is None:
            if problem_type in ['binary_clas', 'multi_clas']:
                eval_metric = 'logloss'
            elif problem_type == 'regression':
                eval_metric = 'mse'
            else:
                raise Exception('Problem type {} not known.'.format(problem_type))

        rt_conf['eval_metric'] = eval_metric

        if target_transform is not None:
            transform_name = target_transform.lower()
            rt_conf['target_transform'] = transforms.target_transform_from_name(transform_name)

        rt_conf['problem_type'] = problem_type
        rt_conf['target'] = target
        rt_conf['exclude_cols'] = exclude_cols

        tree_builder = TreeBuilder(self.config, self.logger)
        self.tree = tree_builder.build()
        self.plots = Plots(rt_conf, self.data, self.tree)
        self.model_results = None
        self.runner = None

    def make_time_series(self, date_split_col, series_key_cols=None, prediction_length=1,
                         frequency=None, n_offset_models=1):
        """
        :param date_split_col: Name of the date column.
        :param series_key_cols: If many time series, these are the keys for the time series
        :param prediction_length: Time steps an individual model should predict into the future
        :param frequency: Frequency of the time series
        :param n_offset_models: Number of models for predicting further into the future
        """
        if date_split_col not in self.data.df.columns:
            raise Exception('The date_split_col {} was not in the columns of the data set {}.'
                            .format(date_split_col, self.data.df.columns))
        if series_key_cols is None:
            series_key_cols = ['guac_time_series_key']
            # this hack makes the whole code that groups by time series key
            # reusable for the case of a single time series.
            # it's omitted in the metadata
            self.data.df['guac_time_series_key'] = True
        elif not isinstance(series_key_cols, list):
            series_key_cols = [series_key_cols]
        for key_col in series_key_cols:
            if key_col not in self.data.df.columns:
                raise Exception('The time series key column {} was not in the'
                                ' columns of the data set {}.'
                                .format(key_col, self.data.df.columns))
        if not isinstance(prediction_length, int) and prediction_length > 0:
            raise Exception('Prediction length must be positive integer, but was {}'
                            .format(prediction_length))

        rt_conf = self.config['run_time']
        ts_conf = rt_conf['time_series']
        rt_conf['is_time_series'] = True
        ts_conf['date_split_col'] = date_split_col
        ts_conf['series_key_cols'] = series_key_cols
        ts_conf['prediction_length'] = prediction_length
        ts_conf['n_offset_models'] = n_offset_models

        if frequency is None:
            frequency = analyze_frequency(self.data.df, ts_conf)
            self.logger.info('Inferred time series frquency of %s', frequency)
        ts_conf['frequency'] = frequency

        # ToDo: this is duplicated from the guac constructor
        tree_builder = TreeBuilder(self.config, self.logger)
        self.tree = tree_builder.build()
        self.plots = Plots(rt_conf, self.data, self.tree)

    def run(self, hyper_param_iterations, prediction_range=None):
        if not isinstance(hyper_param_iterations, int) and hyper_param_iterations > 0:
            raise Exception('Number of hyper parameter iterations must be positive integer,'
                            ' but was {}'.format(hyper_param_iterations))

        # TODO: we shouldn't be mutating config
        self.config['run_time']['hyper_param_iterations'] = hyper_param_iterations
        self.config['run_time']['prediction_range'] = prediction_range

        self.runner = TreeRunner(self.data, self.config, self.tree, self.logger)
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

    def get_pipeline(self, model):
        return Pipeline.from_tree(self, model)
