from guacml.preprocessing.column_analyzer import ColType
from guacml.step_tree.base_step import BaseStep
import pandas as pd


class DateParts(BaseStep):

    def execute_inplace(self, data):
        df = data.df
        meta = data.metadata

        date_cols = meta[meta.type == ColType.DATETIME].index
        for col in date_cols:
            df[col + '_hour'] = df[col].dt.hour
            df[col + '_day'] = df[col].dt.day
            df[col + '_month'] = df[col].dt.month
            df[col + '_year'] = df[col].dt.year
            df[col + '_day_of_week'] = df[col].dt.dayofweek

            new_cols = [
                col + '_hour',
                col + '_day',
                col + '_month',
                col + '_year',
                col + '_day_of_week'
            ]
            to_append = []
            to_append_index = []
            for new_col in new_cols:
                to_append_index.append(new_col)
                to_append.append({
                    'type': ColType.ORDINAL,
                    'derived_from': col,
                    'n_unique': df[new_col].nunique(),
                    'n_na': df[new_col].notnull().sum(),
                    'n_blank': 0
                })
                self.logger.info('DateParts: generated column %s', new_col)

            data.metadata = meta.append(pd.DataFrame(to_append, index=to_append_index))
