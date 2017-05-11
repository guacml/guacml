from .base_feature_generator import BaseFeatureGenerator
from sklearn.preprocessing import OneHotEncoder

class LabelEncoder(BaseFeatureGenerator):
    def generate(self, input):
        enc = OneHotEncoder()
        return enc.fit_transform(input)