from ..base_step import BaseStep
from sklearn.preprocessing import OneHotEncoder as OHE

class OneHotEncoder(BaseStep):

    def __init__(self, column_info):
        self.column_info = column_info

    def execute(self, input):
        enc = OHE()
        return enc.fit_transform(input)
