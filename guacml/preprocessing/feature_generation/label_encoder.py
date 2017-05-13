from ..base_step import BaseStep
from ..column_analyzer import ColType
from sklearn.preprocessing import LabelEncoder as LE

class LabelEncoder(BaseStep):

    def __init__(self, column_info):
        self.column_info = column_info

    def execute(self, input):
        df = input.copy()
        enc = LE()
        info = self.column_info
        to_encode = info[info.type == ColType.CATEGORICAL].col_name
        for col in to_encode:
            try:
                df[col] = enc.fit_transform(df[col])
            except Exception as e:
                print(col)
                raise e
        return df
