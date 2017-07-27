from guacml.enums import ColType
from guacml.step_tree.base_step import BaseStep
import pandas as pd

from guacml.util.time_series_util import analyze_frequency_for_group


class BaseHistoricTarget(BaseStep):
    """
    Base class for columns base on historic target values like medians or simple lagged
    target values.
    """

    def __init__(self, hist_parameters, config, logger, group_keys=None):
        super().__init__(config, logger)
        self.run_time_config = config['run_time']
        self.n_offset_models = self.run_time_config['time_series']['n_offset_models']
        self.hist_parameters = hist_parameters
        if group_keys is not None and \
                not (isinstance(group_keys, list) or isinstance(group_keys, str)):
            raise Exception('Argument group_keys must be an instance of list or string.')
        self.group_keys = group_keys

    def execute_inplace(self, data):
        """
        When we want to predict x time steps into the future, we can not include the last
        x-1 time steps as features, because they will be unknown for the test set.
        But we also want to have recent time steps if we don't predict far in the future,
        as more recent data might be more predictive.
        Solution: Train several models, some for predicting the closer time steps, some
        for the time steps further out. The number of models is called n_offset_models.
        """
        ts_conf = self.run_time_config['time_series']
        date_split_col = ts_conf['date_split_col']
        series_key_cols = ts_conf['series_key_cols']
        prediction_length = ts_conf['prediction_length']
        target = self.run_time_config['target']

        if self.group_keys is None:
            group_keys = series_key_cols
        else:
            if isinstance(self.group_keys, str):
                group_keys = series_key_cols + [self.group_keys]
            else:
                group_keys = series_key_cols + self.group_keys

        df = data.df
        meta = data.metadata

        group_frequency = analyze_frequency_for_group(df, date_split_col, group_keys)
        df = df.reset_index().set_index(date_split_col)
        without_gaps = df.groupby(group_keys)['index', target] \
            .resample(group_frequency) \
            .asfreq() \
            .reset_index(group_keys)

        grouped = without_gaps.sort_index().groupby(group_keys)[target]
        df = df.reset_index().set_index(group_keys + [date_split_col])

        for i_offset in range(self.n_offset_models):
            for hist_param in self.hist_parameters:
                col_name, _ = self.col_name(target, hist_param, i_offset)
                hist_targets = self.calc_hist_target(grouped, group_keys, hist_param)
                if not hist_targets.index.name == date_split_col:
                    raise Exception('The index {} of the hist_targets DataFrame was not the'
                                    ' expected {}. Check if the subclass implements '
                                    'calc_hist_target() correctly.')
                # shift by one step, because the current value must not be leaked
                hist_targets[target] = hist_targets.groupby(group_keys)[target] \
                                                   .shift((i_offset + 1) * prediction_length)
                df[col_name] = hist_targets.reset_index() \
                                           .set_index(group_keys + [date_split_col])

        data.df = df.reset_index().set_index('index')
        to_append = []
        to_append_index = []
        for i_offset in range(self.n_offset_models):
            for hist_param in self.hist_parameters:
                col_name, shared_col_name = self.col_name(target, hist_param, i_offset)
                to_append_index.append(col_name)
                to_append.append({
                    'type': ColType.ORDINAL,
                    'derived_from': target,
                    'n_unique': df[col_name].nunique(),
                    'n_na': df[col_name].notnull().sum(),
                    'n_blank': 0,
                    'is_lagged_target': True,
                    'lagged_target_model_offset': i_offset,
                    'lagged_target_shared_name': shared_col_name
                })
        data.metadata = meta.append(pd.DataFrame(to_append, index=to_append_index))

    def col_name(self, target, hist_param, i_offset):
        if self.group_keys is None:
            shared_name = '{}_{}_{}'.format(target,
                                            self.get_hist_target_name(),
                                            hist_param)
        else:
            shared_name = '{}_{}_{}_by_{}'.format(target,
                                                  self.get_hist_target_name(),
                                                  hist_param,
                                                  self.group_keys)
        if self.n_offset_models == 1:
            return shared_name, shared_name
        else:
            col_name = '{}_offset_{}'.format(shared_name, i_offset)
            return col_name, shared_name

    def get_hist_target_name(self):
        raise NotImplementedError()

    def calc_hist_target(self, grouped_series, group_keys, hist_param):
        """
        Uses historic data to calculate features.

        :param grouped_series: The time series data, possibly grouped
        :param group_keys: Column name for an extra column to group by
        :param hist_param: Several calculations can be done in one go for a set of parameters
        :return: DataFrame indexed by date_split_col that has the historic values
        and group keys as columns
        """
        raise NotImplementedError()
