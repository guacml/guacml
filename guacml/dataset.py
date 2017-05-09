import pandas as pd
from .preprocessing.column_analyzer import ColumnAnalyzer


class Dataset:

    def __init__(self, path, **kwds):
        self.df = pd.read_csv(path, **kwds)

    def analyze_columns(self):
        col_analyzer = ColumnAnalyzer()
        return col_analyzer.analyze(self.df)

    def df(self):
        return self.df
