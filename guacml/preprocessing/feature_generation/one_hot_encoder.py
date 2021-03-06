import pandas as pd

from sklearn.preprocessing import OneHotEncoder as OHE
from guacml.step_tree.base_step import BaseStep


class OneHotEncoder(BaseStep):

    def execute_inplace(self, data):
        meta = data.metadata

        enc = OHE(sparse=False)

        # ToDo: Rather check total number of OneHot columns and define cutoff
        cols_to_encode = meta[
            (meta.type == 'int_encoding') &
            (meta.n_unique > 2) &
            (meta.n_unique < self.config['pre_processing']['max_uniques_for_one_hot'])
        ].index

        for col in cols_to_encode:
            new_cols = enc.fit_transform(data.df[[col]])
            new_cols = pd.DataFrame(new_cols, index=data.df.index)
            new_col_names = [col + '_one_hot_' + str(i) for i in enc.active_features_]
            new_cols.columns = new_col_names
            data.df = pd.concat([data.df, new_cols], axis=1)

            n_new = len(new_col_names)
            to_append = pd.DataFrame({
                'type': ['binary'] * n_new,
                'derived_from': [col] * n_new,
                'n_unique': [2] * n_new,
                'n_na': [0] * n_new,
                'n_blank': [0] * n_new
            }, index=new_col_names)
            self.logger.info('OneHotEncoder: generated %d columns from %s', n_new, col)
            data.metadata = meta.append(to_append)
