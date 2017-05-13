from .base_step import BaseStep

class FillNa(BaseStep):

    def __init__(self, column_info):
        self.column_info = column_info

    def execute(self, input):
        df = input.copy()
        info = self.column_info
        col_to_fill = info[info.n_na > 0].col_name
        return df[col_to_fill].fillna(inplace=True)
