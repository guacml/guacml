from guacml.step_tree.base_step import BaseStep


class FillNa(BaseStep):

    def execute(self, input, metadata):
        df = input.copy()
        meta = metadata.copy()

        col_to_fill = metadata[metadata.n_na > 0].col_name
        # ToDo: Better fill method
        df[col_to_fill] = df[col_to_fill].fillna(df[col_to_fill].mean())
        meta.loc[meta.col_name.isin(col_to_fill), 'n_na'] = 0
        return df, meta
