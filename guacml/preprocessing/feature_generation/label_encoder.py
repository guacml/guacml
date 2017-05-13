from ..base_step import BaseStep
from sklearn.preprocessing import LabelEncoder as LE

class LabelEncoder(BaseStep):
    def execute(self, input):
        enc = LE()
        return enc.fit_transform(input)
