import pandas as pd
from .preprocessing.column_analyzer import ColumnAnalyzer
from IPython.display import clear_output
from .preprocessing.tree_builder import TreeBuilder
from .preprocessing.tree_runner import TreeRunner

class Dataset:

    def __init__(self, path, target, **kwds):
        print('loading data..')
        self.df = pd.read_csv(path, **kwds)
        if not target in self.df.columns:
            raise ValueError('The target {0} does not exist as column.\n'
                             'Available columns: {1}'.format(target, self.df.columns))
        self.target = target
        print('analyzing columns..')
        col_analyzer = ColumnAnalyzer()
        self.column_info = col_analyzer.analyze(self.df)
        clear_output()


    def run(self):
        tree_builder = TreeBuilder(self.column_info)
        tree = tree_builder.build()

        runner = TreeRunner(tree)
        runner.run(self.df)