from guacml.preprocessing.column_analyzer import ColType
from ..base_step import BaseStep
from sklearn.preprocessing import OneHotEncoder as OHE
import pandas as pd

class OneHotEncoder(BaseStep):

    def execute(self, input, metadata):
        df = input.copy()
        meta = metadata.copy()

        enc = OHE(sparse=False)

        # ToDo: Rather check total number of OneHot columns and define cutoff
        cols_to_encode = meta[(meta.type == ColType.INT_ENCODING) &
                              (meta.n_unique < 50)].col_name

        for col in cols_to_encode:
            new_cols = enc.fit_transform(df[[col]])
            new_cols = pd.DataFrame(new_cols)
            new_col_names = [col + '_one_hot_' + str(i) for i in enc.active_features_]
            new_cols.columns = new_col_names
            df = pd.concat([df, new_cols], axis=1)

            n_new = len(new_col_names)
            to_append = pd.DataFrame({
                'col_name': new_col_names,
                'type': [ColType.ONE_HOT_ENCODING] * n_new,
                'derived_from': [col] * n_new,
                'n_unique': [2] * n_new,
                'n_na': [0] * n_new,
                'n_blank': [0] * n_new
            })
            meta = meta.append(to_append)

        return df, meta
