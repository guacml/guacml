import pandas as pd

from guacml.enums import ProblemType
from guacml.metrics.accuracy import Accuracy
from guacml.metrics.log_loss import LogLoss
from guacml.metrics.mean_squared_error import MeanSquaredError
from guacml.metrics.root_mean_squared_log_error import RootMeanSquaredLogError
from guacml.plots import Plots
from guacml.step_tree.random_splitter import RandomSplitter
from guacml.step_tree.step_tree import StepTree
from guacml.step_tree.tree_builder import TreeBuilder
from guacml.step_tree.tree_runner import TreeRunner
from .preprocessing.column_analyzer import ColumnAnalyzer, ColType
from IPython.display import clear_output


class Dataset:

    def __init__(self, path, target, exclude_cols=None, eval_metric=None, **kwds):
        print('loading data..')
        self.df = pd.read_csv(path, **kwds)
        if target not in self.df.columns:
            raise ValueError('The target {0} does not exist as column.\n'
                             'Available columns: {1}'.format(target, self.df.columns))
        if exclude_cols is not None:
            for col in exclude_cols:
                if not col in self.df.columns:
                    raise ValueError('The column to exclude {0} does not exist as column.\n'
                                     'Available columns: {1}'.format(target, self.df.columns))
                del self.df[col]

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

        if eval_metric is not None:
            if eval_metric.lower() == 'accuracy':
                self.eval_metric = Accuracy()
            elif eval_metric.lower() == 'rmsle':
                self.eval_metric = RootMeanSquaredLogError()
            else:
                raise NotImplementedError('Unknown eval metic: ' + eval_metric)

        self.target = target
        self.plots = Plots(self.problem_type)
        self.splitter = RandomSplitter(.8)
        self.model_results = None

    def run(self, hyper_param_iterations):
        tree_builder = TreeBuilder(self.problem_type)
        step_tree = StepTree(self.target, hyper_param_iterations, self.eval_metric)
        tree = tree_builder.build(step_tree)

        runner = TreeRunner(self, tree)
        self.model_results = runner.run()
        self.plots.set_model_results(self.model_results)

    def model_overview(self):
        rows = []
        for name, res in self.model_results.items():
            res_dict = res.to_display_dict()
            res_dict['model name'] = name
            rows.append(res_dict)

        columns = ['model name', 'n features', 'holdout error', 'holdout error interval', 'cv error', 'training error']
        result = pd.DataFrame(rows,
                              columns=columns + ['holdout error numeric'])
        return result.sort_values('holdout error numeric')[columns]

    def hyper_param_runs(self, model_name):
        if model_name in self.model_results:
            return self.model_results[model_name].all_hyper_param_runs
        else:
            raise ValueError('Model name has to be in {0}'.format(self.model_results.keys()))
