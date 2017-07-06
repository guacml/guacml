from guacml.step_tree.base_step import BaseStep


class ColumnCleaner(BaseStep):
    def execute(self, data):
        message = 'Replaced %d blank values with None for column %s:.'

        data = data.copy()
        for _, col_desc in data.metadata.iterrows():
            if col_desc.n_blank > 0:
                col = col_desc.col_name
                data.df[col].replace('', None, inplace=True)
                col_desc.n_blank = 0
                self.logger.info(message, col_desc.n_blanks, col)

        return data
