from guacml.step_tree.base_step import BaseStep


class FillNa(BaseStep):

    def execute(self, dataframe, metadata):
        df_copy = dataframe.copy()
        meta = metadata.copy()

        col_to_fill = metadata[metadata.n_na > 0].col_name
        # ToDo: Better fill method
        df_copy[col_to_fill] = df_copy[col_to_fill].fillna(df_copy[col_to_fill].mean())
        meta.loc[meta.col_name.isin(col_to_fill), 'n_na'] = 0
        return df_copy, meta
