import pandas as pd

from guacml.dataset import Dataset
from guacml.preprocessing.column_analyzer import ColType
from sklearn.preprocessing import OneHotEncoder as OHE
from guacml.step_tree.base_step import BaseStep


class OneHotEncoder(BaseStep):
    def __init__(self, preprocess_config):
        self.pre_process_config = preprocess_config

    def execute(self, data):
        data = data.copy()
        meta = data.metadata

        enc = OHE(sparse=False)

        # ToDo: Rather check total number of OneHot columns and define cutoff
        cols_to_encode = meta[(meta.type == ColType.INT_ENCODING) &
                              (meta.n_unique < self.pre_process_config['max_uniques_for_one_hot'])].col_name

        for col in cols_to_encode:
            new_cols = enc.fit_transform(data.df[[col]])
            new_cols = pd.DataFrame(new_cols)
            new_col_names = [col + '_one_hot_' + str(i) for i in enc.active_features_]
            new_cols.columns = new_col_names
            data.df = pd.concat([data.df, new_cols], axis=1)

            n_new = len(new_col_names)
            to_append = pd.DataFrame({
                'col_name': new_col_names,
                'type': [ColType.BINARY] * n_new,
                'derived_from': [col] * n_new,
                'n_unique': [2] * n_new,
                'n_na': [0] * n_new,
                'n_blank': [0] * n_new
            })
            data.metadata = meta.append(to_append)

        return data
