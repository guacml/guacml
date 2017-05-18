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
        self.splitter = RandomSplitter(.8)

    def run(self, hyper_param_iterations):
        tree_builder = TreeBuilder(self.metadata, self.target)
        tree = tree_builder.build(hyper_param_iterations)

        runner = TreeRunner(self, tree)
        self.model_results = runner.run()

    def model_overview(self):
        rows = []
        for name, res in self.model_results.items():
            res_dict = res.to_display_dict()
            res_dict['model name'] = name
            rows.append(res_dict)
        result = pd.DataFrame(rows, columns=['model name', 'holdout error', 'cv error', 'training error'])
        return result.sort_values('holdout error')

    def hyper_param_runs(self, model_name):
        if model_name in self.model_results:
            return self.model_results[model_name].all_hyper_param_runs
        else:
            raise ValueError('Model name has to be in {0}'.format(self.model_results.keys()))

    def error_overview(self, model_name):
        return self.model_results[model_name].holdout_row_errors.hist()