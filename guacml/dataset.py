import pandas as pd

class Dataset:
    def __init__(self, path):
        self.df = pd.read_csv(path)
