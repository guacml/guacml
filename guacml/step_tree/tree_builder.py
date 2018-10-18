from guacml.models.linear_model import LinearModel
from guacml.models.random_forest import RandomForest
from guacml.models.xgboost import XGBoost
from guacml.preprocessing.column_cleaner import ColumnCleaner
from guacml.preprocessing.feature_whitelist import FeatureWhitelist
from guacml.preprocessing.feature_generation.date_parts import DateParts
from guacml.preprocessing.feature_generation.historical_medians import HistoricalMedians
from guacml.preprocessing.feature_generation.label_encoder import LabelEncoder
from guacml.preprocessing.feature_generation.one_hot_encoder import OneHotEncoder
from guacml.preprocessing.fill_na import FillNa
from guacml.step_tree.step_tree import StepTree


class TreeBuilder:

    def __init__(self, config, logger):
        self.config = config
        self.logger = logger

    def build(self):
        step_tree = StepTree(self.config, self.logger)
        step_tree.add_step('clean_columns', None, ColumnCleaner(self.config, self.logger))
        step_tree.add_step('encode_labels', 'clean_columns', LabelEncoder(self.config, self.logger))
        last_node = 'date_parts'
        step_tree.add_step('date_parts', 'encode_labels', DateParts(self.config, self.logger))

        if self.config['run_time']['is_time_series']:
            step_tree.add_step('historical_medians',
                               'date_parts',
                               HistoricalMedians([1, 5, 20], self.config, self.logger))
            last_node = 'historical_medians'

        step_tree.add_step('feature_whitelist', last_node,
                           FeatureWhitelist(self.config, self.logger))
        step_tree.add_step('fill_na', 'feature_whitelist', FillNa(self.config, self.logger))

        step_tree.add_model('xgboost', 'feature_whitelist', XGBoost(self.config, self.logger))

        step_tree.add_model('random_forest', 'fill_na', RandomForest(self.config, self.logger))

        step_tree.add_step('one_hot_encode', 'fill_na', OneHotEncoder(self.config, self.logger))
        step_tree.add_model('linear_model', 'one_hot_encode', LinearModel(self.config, self.logger))

        return step_tree
