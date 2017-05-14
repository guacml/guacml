from guacml.step_tree.base_step import BaseStep


class ColumnCleaner(BaseStep):
    def execute(self, input, metadata):
        for _, col_desc in metadata.iterrows():
            if col_desc.n_blank > 0:
                col = col_desc.col_name
                col = col.replace('', None)
                col_desc.n_blank = 0
                print('Replaced {0} blank values with None for column {1}:.'.format(col_desc.n_blanks, col))

        return input, metadata