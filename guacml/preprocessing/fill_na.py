from guacml.step_tree.base_step import BaseStep


class FillNa(BaseStep):

    def execute(self, data):
        data = data.copy()
        df = data.df
        meta = data.metadata

        col_to_fill = meta[meta.n_na > 0].index
        # ToDo: Better fill method, depending on column type
        df[col_to_fill] = df[col_to_fill].fillna(df[col_to_fill].mean())
        # in case the mean is also N/A
        df[col_to_fill] = df[col_to_fill].fillna(0)
        meta.loc[meta.index.isin(col_to_fill), 'n_na'] = 0

        return data
