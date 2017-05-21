import pandas as pd

from guacml.enums import ProblemType
from guacml.metrics.accuracy import Accuracy
from guacml.metrics.log_loss import LogLoss
from guacml.metrics.mean_squared_error import MeanSquaredError
from guacml.plots import Plots
from guacml.step_tree.random_splitter import RandomSplitter
from guacml.step_tree.step_tree import StepTree
from guacml.step_tree.tree_builder import TreeBuilder
from guacml.step_tree.tree_runner import TreeRunner
from .preprocessing.column_analyzer import ColumnAnalyzer, ColType
from IPython.display import clear_output


class Dataset:

    def __init__(self, path, target, eval_metric=None, **kwds):
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

        target_meta = self.metadata[self.metadata.col_name == target]
        if target_meta.iloc[0].type == ColType.BINARY:
            self.problem_type = ProblemType.BINARY_CLAS
            self.eval_metric = LogLoss()
            print('Binary classification problem detected.')
        elif target_meta.iloc[0].type in [ColType.CATEGORICAL, ColType.INT_ENCODING]:
            self.problem_type = ProblemType.MULTI_CLAS
            self.eval_metric = LogLoss()
            print('Multi class classification problem detected.')
        elif target_meta.iloc[0].type in [ColType.ORDINAL, ColType.NUMERIC]:
            self.problem_type = ProblemType.REGRESSION
            self.eval_metric = MeanSquaredError()
            print('Regression problem detected.')
        else:
            print('Can not automatically infer problem type.')

        if not eval_metric is None:
            if eval_metric.lower() == 'accuracy':
                self.eval_metric = Accuracy()
            else:
                raise NotImplementedError('Unknown eval metic: ' + eval_metric)

        self.splitter = RandomSplitter(.8)

    def run(self, hyper_param_iterations):
        tree_builder = TreeBuilder(self.problem_type)
        step_tree = StepTree(self.target, hyper_param_iterations, self.eval_metric)
        tree = tree_builder.build(step_tree)

        runner = TreeRunner(self, tree)
        self.model_results = runner.run()
        self.plots = Plots(self.model_results)

    def model_overview(self):
        rows = []
        for name, res in self.model_results.items():
            res_dict = res.to_display_dict()
            res_dict['model name'] = name
            rows.append(res_dict)

        result = pd.DataFrame(rows,
                              columns=['model name', 'holdout error', 'cv error', 'training error'])
        # sorted as strings
        return result.sort_values('holdout error', ascending=False)

    def hyper_param_runs(self, model_name):
        if model_name in self.model_results:
            return self.model_results[model_name].all_hyper_param_runs
        else:
            raise ValueError('Model name has to be in {0}'.format(self.model_results.keys()))
