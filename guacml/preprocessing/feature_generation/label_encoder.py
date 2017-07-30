from guacml.step_tree.base_step import BaseStep
from sklearn.preprocessing import LabelEncoder as LE


class LabelEncoder(BaseStep):

    def execute_inplace(self, data):
        df = data.df
        meta = data.metadata
        needs_fitting = self.state is None
        classes = {} if needs_fitting else self.state['classes']
        cols_to_encode = meta[meta.type == 'categorical'].index if needs_fitting else list(classes)

        for col in cols_to_encode:
            encoder = LE()
            input = df.loc[df[col].notnull(), col]

            if needs_fitting:
                encoder.fit(input)
                classes[col] = encoder.classes_.tolist()
            else:
                encoder.classes_ = classes[col]

            df.loc[df[col].notnull(), col] = encoder.transform(input)
            df[col] = df[col].astype(float)
            meta.loc[col, 'type'] = 'int_encoding'
            meta.loc[col, 'derived_from'] = col
            self.logger.info('LabelEncoder: encoded %s', col)

        if needs_fitting:
            self.state = {'classes': classes}
