from guacml.models.linear_model import LinearModel
from guacml.models.random_forest import RandomForest
from guacml.models.xgboost import XgBoost
from guacml.preprocessing.column_cleaner import ColumnCleaner
from guacml.preprocessing.feature_generation.date_splitter import DateSplitter
from guacml.preprocessing.feature_generation.label_encoder import LabelEncoder
from guacml.preprocessing.feature_generation.one_hot_encoder import OneHotEncoder
from guacml.preprocessing.fill_na import FillNa
from guacml.step_tree.step_tree import StepTree


class TreeBuilder:
    def __init__(self, config):
        self.config = config
        self.problem_type = config['run_time']['problem_type']

    def build(self):
        step_tree = StepTree(self.config)
        step_tree.add_step('clean_columns', None, ColumnCleaner())
        step_tree.add_step('encode_labels', 'clean_columns', LabelEncoder())
        step_tree.add_step('split_dates', 'encode_labels', DateSplitter())
        step_tree.add_model('xg_boost', 'split_dates', XgBoost(self.problem_type))

        step_tree.add_step('fill_na', 'split_dates', FillNa())
        step_tree.add_model('random_forest', 'fill_na', RandomForest(self.problem_type))

        step_tree.add_step('one_hot_encode', 'fill_na',
                           OneHotEncoder(self.config['pre_processing']))
        step_tree.add_model('linear_model', 'one_hot_encode', LinearModel(self.problem_type))

        return step_tree
