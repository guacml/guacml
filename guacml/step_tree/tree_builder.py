from guacml.preprocessing.column_cleaner import ColumnCleaner
from .step_tree import StepTree
from ..preprocessing.feature_generation.label_encoder import LabelEncoder
from ..preprocessing.feature_generation.one_hot_encoder import OneHotEncoder
from ..preprocessing.fill_na import FillNa
from ..models.xgboost import XgBoost
from ..models.random_forest import RandomForest
from ..models.linear_model import LinearModel


class TreeBuilder:
    def __init__(self, problem_type):
        self.problem_type = problem_type

    def build(self, step_tree):
        step_tree.add_step('clean_columns', None, ColumnCleaner())
        step_tree.add_step('encode_labels', 'clean_columns', LabelEncoder())
        step_tree.add_model('xg_boost', 'encode_labels', XgBoost(self.problem_type))

        step_tree.add_step('fill_na', 'encode_labels', FillNa())
        step_tree.add_model('random_forest', 'fill_na', RandomForest(self.problem_type))

        step_tree.add_step('one_hot_encode', 'fill_na', OneHotEncoder())
        step_tree.add_model('linear_model', 'one_hot_encode', LinearModel(self.problem_type))
        return step_tree
