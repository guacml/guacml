from .step_tree import StepTree
from .feature_generation.label_encoder import LabelEncoder
from .feature_generation.one_hot_encoder import OneHotEncoder
from .fill_na import FillNa
from ..models.xgboost import XgBoost
from ..models.random_forest import RandomForest
from ..models.linear_model import LinearModel

class TreeBuilder:
    def __init__(self, column_info):
        self.column_info = column_info

    def build(self):
        step_tree = StepTree()
        step_tree.add_step('encode_labels', None, LabelEncoder(column_info))
        step_tree.add_step('xg_boost', 'encode_labels', XgBoost())

        step_tree.add_step('fill_na', 'encode_labels', FillNa(column_info))
        step_tree.add_step('random_forest', 'fill_na', RandomForest())

        step_tree.add_step('one_hot_encode', 'fill_na', OneHotEncoder(column_info))
        step_tree.add_step('linear_model', 'one_hot_encode', LinearModel())
        return step_tree
