from guacml.step_tree.base_step import BaseStep


class ColumnCleaner(BaseStep):
    def execute_inplace(self, data):
        for _, col_desc in data.metadata.iterrows():
            if col_desc.n_blank > 0:
                col = col_desc.col_name
                data.df[col].replace('', None, inplace=True)
                col_desc.n_blank = 0
                self.logger.info('Replaced %d blank values with None for column %s',
                                 col_desc.n_blanks, col)
