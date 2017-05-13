from ..base_step import BaseStep
from sklearn.preprocessing import OneHotEncoder as OHE

class OneHotEncoder(BaseStep):

    def execute(self, input, metadata):
        enc = OHE()
        return input, metadata
