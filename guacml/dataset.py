import pandas as pd
from .preprocessing.column_analyzer import ColumnAnalyzer
from collections import Iterable
from IPython.display import clear_output

class Dataset:

    def __init__(self, path, target, **kwds):
        print('loading data..')
        self.df = pd.read_csv(path, **kwds)
        self.check_target(target)
        self.target = target
        print('analyzing columns..')
        col_analyzer = ColumnAnalyzer()
        self.column_info = col_analyzer.analyze(self.df)
        clear_output()

    def check_target(self, target):
        missing = []
        if isinstance(target, Iterable) and not isinstance(target, str):
            for col in target:
                if not col in self.df.columns:
                    missing.append(col)
        elif not target in self.df.columns:
            missing.append(target)

        if len(missing) > 1:
            raise ValueError('The following targets do not exist as columns: {0}\n'
                             'Available columns: {1}'.format(missing, self.df.columns))
        elif len(missing) == 1:
            raise ValueError('The following target does not exist as column: {0}\n'
                             'Available columns: {1}'.format(missing, self.df.columns))

    def df(self):
        return self.df
