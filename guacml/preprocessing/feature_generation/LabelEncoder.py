from .base_feature_generator import BaseFeatureGenerator
from sklearn.preprocessing import LabelEncoder

class LabelEncoder(BaseFeatureGenerator):
    def generate(self, input):
        enc = LabelEncoder()
        return enc.fit_transform(input)