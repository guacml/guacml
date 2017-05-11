from .step_dag import StepDag

class FeaturePipeline:
    def __init__(self):
        self.step_dag = StepDag()

    def run(self, df_raw):
        feats_xgboost = xgboost_features(df_raw)
        feats_random_forest = random_forest_features(feats_xgboost)
        feats_linear_model = linear_model_features(feats_xgboost)