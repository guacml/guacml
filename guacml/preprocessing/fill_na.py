from guacml.step_tree.base_step import BaseStep


class FillNa(BaseStep):

    def execute(self, data):
        data = data.copy()
        df = data.df
        meta = data.metadata

        col_to_fill = meta[meta.n_na > 0].col_name
        # ToDo: Better fill method
        df[col_to_fill] = df[col_to_fill].fillna(df[col_to_fill].mean())
        meta.loc[meta.col_name.isin(col_to_fill), 'n_na'] = 0
        return data
