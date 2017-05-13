from .step_tree import StepTree
from .feature_generation import LabelEncoder, OneHotEncoder

class TreeBuilder:
    def __init__(self, column_info):
        self.column_info = column_info

    def build(self):
        step_tree = StepTree()
        step_tree.add_step('encode_labels', None, LabelEncoder())
        step_tree.add_step('xg_boost', 'encode_labels', XgBoost())

        step_tree.add_step('fill_na', 'encode_labels', FillNa())
        step_tree.add_step('random_forest', 'fill_na', RandomForest())

        step_tree.add_step('one_hot_encode', 'fill_na', OneHotEncoder())
        step_tree.add_step('linear_model', 'one_hot_encode', LinearModel())
        return step_tree