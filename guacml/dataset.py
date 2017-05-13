import pandas as pd

from guacml.step_tree.random_splitter import RandomSplitter
from guacml.step_tree.tree_builder import TreeBuilder
from guacml.step_tree.tree_runner import TreeRunner
from .preprocessing.column_analyzer import ColumnAnalyzer
from IPython.display import clear_output



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
        self.column_info, self.metadata = col_analyzer.analyze(self.df)
        clear_output()
        self.splitter = RandomSplitter(.7)

    def run(self):
        tree_builder = TreeBuilder(self.metadata, self.target)
        tree = tree_builder.build()

        runner = TreeRunner(self, tree)
        return runner.run()
