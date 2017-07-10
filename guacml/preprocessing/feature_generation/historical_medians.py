from guacml.enums import ColType
from guacml.step_tree.base_step import BaseStep
import pandas as pd


class HistoricalMedians(BaseStep):

    def execute_inplace(self, data):
        run_time_config = self.config['run_time']
        date_split_col = run_time_config['time_series']['date_split_col']
        series_key_cols = run_time_config['time_series']['series_key_cols']
        prediction_length = run_time_config['time_series']['prediction_length']
        target = run_time_config['target']
        df = data.df
        meta = data.metadata

        df = df.reset_index().set_index(date_split_col).sort_index()
        grouped = df.groupby(series_key_cols)[target]
        df = df.reset_index().set_index(series_key_cols + [date_split_col]).sort_index()

        col_names = {}
        windows = [prediction_length, 5 * prediction_length, 20 * prediction_length]
        for window in windows:
            col_name = 'median_{0}'.format(window)
            df[col_name] = grouped.rolling(window).median().shift(1)
            col_names[window] = col_name

        data.df = df.reset_index().set_index('index')
        to_append = []
        to_append_index = []
        for window in windows:
            col_name = col_names[window]
            to_append_index.append(col_name)
            to_append.append({
                'type': ColType.ORDINAL,
                'derived_from': target,
                'n_unique': df[col_name].nunique(),
                'n_na': df[col_name].notnull().sum(),
                'n_blank': 0
            })
        data.metadata = meta.append(pd.DataFrame(to_append, index=to_append_index))
