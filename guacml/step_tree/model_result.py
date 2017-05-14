class ModelResult:

    def __init__(self, model, training_error, cv_error, cv_predictions):
        self.model = model
        self.training_error = training_error
        self.cv_error = cv_error
        self.cv_predictions = cv_predictions

    def to_display_dict(self):
        return {
            'cv error': '{:.4g}'.format(self.cv_error),
            'training error': '{:.4g}'.format(self.training_error)
        }
