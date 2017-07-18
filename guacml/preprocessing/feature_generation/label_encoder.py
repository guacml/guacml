from guacml.step_tree.base_step import BaseStep
from guacml.preprocessing.column_analyzer import ColType
from sklearn.preprocessing import LabelEncoder as LE


class LabelEncoder(BaseStep):

    def execute_inplace(self, data):
        df = data.df
        meta = data.metadata

        enc = LE()
        cols_to_encode = meta[meta.type == ColType.CATEGORICAL].index
        for col in cols_to_encode:
            df.loc[df[col].notnull(), col] = enc.fit_transform(df.loc[df[col].notnull(), col])
            df[col] = df[col].astype(float)
            meta.loc[col, 'type'] = ColType.INT_ENCODING
            meta.loc[col, 'derived_from'] = col
            self.logger.info('LabelEncoder: encoded %s', col)
