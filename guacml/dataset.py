import pandas as pd

from .preprocessing.column_analyzer import ColumnAnalyzer, ColType
from IPython.display import clear_output


class Dataset:
    @staticmethod
    def read_csv(data_path, target, exclude_cols, **kwds):
        print('loading data..')
        df = pd.read_csv(data_path, **kwds)
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

        print('analyzing columns..')
        col_analyzer = ColumnAnalyzer()
        metadata = col_analyzer.analyze(df)
        clear_output()

        return Dataset(df, metadata)

    def __init__(self, df, metadata):
        self.df = df
        self.metadata = metadata

    def copy(self):
        return Dataset(self.df.copy(), self.metadata.copy())

    def display_metadata(self):
        meta = self.metadata.copy()
        meta['n_unique_%'] = (meta['n_unique'] / meta.shape[0]).round()
        meta['n_na_%'] = (meta['n_na'] / meta.shape[0]).round()
        meta['n_blank_%'] = (meta['n_blank'] / meta.shape[0]).round()

        return meta[['col_name', 'type', 'n_unique', 'n_unique_%', 'n_na_%', 'n_blank_%', 'example']]


