import pandas as pd
from guacml.preprocessing.column_analyzer import ColumnAnalyzer
from IPython.display import clear_output
import joblib

class Dataset:
    @staticmethod
    def from_df(df, target, exclude_cols, **kwds):
        print('loading data..')
        print('finished loading data')

        if exclude_cols is not None:
            for col in exclude_cols:
                if not col in df.columns:
                    raise ValueError('The column to exclude {0} does not exist as column.\n'
                                     'Available columns: {1}'.format(col, df.columns))
                del df[col]

        if target not in df.columns:
            raise ValueError('The target {0} does not exist as column.\n'
                             'Available columns: {1}'.format(target, df.columns))

        df_hash = joblib.hash(df)
        metadata = Dataset.get_metadata(df)

        return Dataset(df, metadata, df_hash)

    @staticmethod
    def get_metadata(df):
        print('analyzing columns..')
        col_analyzer = ColumnAnalyzer()
        metadata = col_analyzer.analyze(df)
        clear_output()
        return metadata

    def __init__(self, df, metadata, data_path=None, df_hash=None):
        self.df = df
        self.metadata = metadata
        self.df_hash = df_hash

    def copy(self):
        return Dataset(self.df.copy(), self.metadata.copy())

    def display_metadata(self):
        meta = self.metadata.copy()
        n_rows = self.df.shape[0]
        meta['n_unique_%'] = (meta['n_unique'] / n_rows).round()
        meta['n_na_%'] = (meta['n_na'] / n_rows).round()
        meta['n_blank_%'] = (meta['n_blank'] / n_rows).round()

        return meta[['type', 'n_unique', 'n_unique_%', 'n_na_%', 'n_blank_%', 'example']]


