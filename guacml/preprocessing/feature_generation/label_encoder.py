from guacml.step_tree.base_step import BaseStep
from ..column_analyzer import ColType
from sklearn.preprocessing import LabelEncoder as LE


class LabelEncoder(BaseStep):

    def execute(self, data):
        data = data.copy()
        df = data.df
        meta = data.metadata

        enc = LE()
        cols_to_encode = meta[meta.type == ColType.CATEGORICAL].col_name
        for col in cols_to_encode:
            df.loc[df[col].notnull(), col] = enc.fit_transform(df.loc[df[col].notnull(), col])
            df[col] = df[col].astype(float)
            meta.loc[meta.col_name == col, 'type'] = ColType.INT_ENCODING
            meta.loc[meta.col_name == col, 'derived_from'] = col
        return data
