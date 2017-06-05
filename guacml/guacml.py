import os
import yaml
import pandas as pd

from guacml.dataset import Dataset
from guacml.enums import ProblemType, ColType
from guacml.metrics.accuracy import Accuracy
from guacml.metrics.log_loss import LogLoss
from guacml.metrics.mean_squared_error import MeanSquaredError
from guacml.metrics.root_mean_squared_log_error import RootMeanSquaredLogError
from guacml.plots import Plots

from guacml.step_tree.step_tree import StepTree
from guacml.step_tree.tree_builder import TreeBuilder
from guacml.step_tree.tree_runner import TreeRunner


class GuacMl:

    def __init__(self, path, target, exclude_cols=None, eval_metric=None, **kwds):
        conf_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
        with open(conf_path, 'r') as file:
            self.config = yaml.load(file)

        self.data = Dataset.read_csv(path, target, exclude_cols, **kwds)

        metadata = self.data.metadata
        target_meta = metadata[metadata.col_name == target]
        rt_conf = self.config['run_time']
        if target_meta.iloc[0].type == ColType.BINARY:
            problem_type = ProblemType.BINARY_CLAS
            rt_conf['eval_metric'] = LogLoss()
            print('Binary classification problem detected.')
        elif target_meta.iloc[0].type in [ColType.CATEGORICAL, ColType.INT_ENCODING]:
            problem_type = ProblemType.MULTI_CLAS
            rt_conf['eval_metric'] = LogLoss()
            print('Multi class classification problem detected.')
        elif target_meta.iloc[0].type in [ColType.ORDINAL, ColType.NUMERIC]:
            problem_type = ProblemType.REGRESSION
            rt_conf['eval_metric'] = MeanSquaredError()
            print('Regression problem detected.')
        else:
            raise Exception('Can not automatically infer problem type.')

        if eval_metric is not None:
            if eval_metric.lower() == 'accuracy':
                rt_conf['eval_metric'] = Accuracy()
            elif eval_metric.lower() == 'rmsle':
                rt_conf['eval_metric'] = RootMeanSquaredLogError()
            else:
                raise NotImplementedError('Unknown eval metric: ' + eval_metric)

        rt_conf['problem_type'] = problem_type
        rt_conf['target'] = target
        rt_conf['exclude_cols'] = exclude_cols
        self.plots = Plots(rt_conf, self.data)
        self.model_results = None

    def run(self, min_hyper_param_iterations):

        tree_builder = TreeBuilder(self.config)
        step_tree = StepTree(self.config)
        tree = tree_builder.build(step_tree)

        runner = TreeRunner(self.data, tree)
        self.model_results = runner.run(min_hyper_param_iterations)
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
