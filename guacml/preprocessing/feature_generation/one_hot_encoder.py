from ..base_step import BaseStep
from sklearn.preprocessing import OneHotEncoder as OHE

class OneHotEncoder(BaseStep):
    def execute(self, input):
        enc = OHE()
        return enc.fit_transform(input)
