from guacml.enums import ColType
from guacml.step_tree.base_step import BaseStep
import pandas as pd


class HistoricalMedians(BaseStep):

    def __init__(self, config):
        super().__init__(config)
        self.run_time_config = config['run_time']
        self.n_offset_models = self.run_time_config['time_series']['n_offset_models']

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

        df = data.df
        meta = data.metadata

        df = df.reset_index().set_index(date_split_col).sort_index()
        grouped = df.groupby(series_key_cols)[target]
        df = df.reset_index().set_index(series_key_cols + [date_split_col]).sort_index()

        median_windows = [prediction_length, 5 * prediction_length, 20 * prediction_length]
        for i_offset in range(self.n_offset_models):
            for median_win in median_windows:
                col_name, _ = self.col_name(target, median_win, i_offset)
                df[col_name] = grouped.rolling(median_win).median().shift((i_offset + 1) * prediction_length)

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
        shared_name = '{}_median_{}'.format(target, median_win)
        if self.n_offset_models == 1:
            return shared_name, shared_name
        else:
            col_name = '{}_offset_{}'.format(shared_name, i_offset)
            return col_name, shared_name

