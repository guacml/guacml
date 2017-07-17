from guacml.enums import ColType
from guacml.step_tree.base_step import BaseStep
import pandas as pd

from guacml.util.time_series_util import analyze_frequency_for_group


class HistoricalMedians(BaseStep):

    def __init__(self, hist_length_factors, config, group_keys=None):
        super().__init__(config)
        self.run_time_config = config['run_time']
        self.n_offset_models = self.run_time_config['time_series']['n_offset_models']
        self.hist_length_factors = hist_length_factors
        if group_keys is not None and\
           not (isinstance(group_keys, list) or isinstance(group_keys, str)):
            raise Exception('Argument group_keys must be an instance of list or string.')
        self.group_keys = group_keys

    def execute_inplace(self, data):
        """
        When we want to predict x time steps into the future, we can not include the last
        x-1 time steps into the median, because they will be unknown for the test set.
        But we also want to have recent time steps if we don't predict far in the future,
        as they make prediction better.
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
        without_gaps = df.groupby(group_keys)['index', target]\
                         .resample(group_frequency)\
                         .asfreq()\
                         .reset_index(group_keys)

        grouped = without_gaps.sort_index().groupby(group_keys)[target]
        df = df.reset_index().set_index(group_keys + [date_split_col])
        median_windows = [length_factor * prediction_length
                          for length_factor in self.hist_length_factors]
        for i_offset in range(self.n_offset_models):
            for median_win in median_windows:
                col_name, _ = self.col_name(target, median_win, i_offset)
                # ToDo: replace the rolling and shift here with a DateOffset
                # ToDo: (already tried for some hours, but lead to much trouble)
                medians = grouped.rolling(median_win, min_periods=1)\
                                 .median()\
                                 .reset_index(group_keys)
                medians[target] = medians.groupby(group_keys)[target]\
                                         .shift((i_offset + 1) * prediction_length)
                df[col_name] = medians.reset_index()\
                                      .set_index(group_keys + [date_split_col])

        data.df = df.reset_index().set_index('index')
        to_append = []
        to_append_index = []
        for i_offset in range(self.n_offset_models):
            for median_win in median_windows:
                col_name, shared_col_name = self.col_name(target, median_win, i_offset)
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

    def col_name(self, target, median_win, i_offset):
        if self.group_keys is None:
            shared_name = '{}_median_{}'.format(target, median_win)
        else:
            shared_name = '{}_median_{}_by_{}'.format(target, median_win, self.group_keys)
        if self.n_offset_models == 1:
            return shared_name, shared_name
        else:
            col_name = '{}_offset_{}'.format(shared_name, i_offset)
            return col_name, shared_name
