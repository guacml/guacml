class ModelResult:

    def __init__(self,
                 model,
                 target,
                 training_error,
                 cv_error,
                 holdout_error,
                 holdout_data,
                 metadata,
                 hyper_params,
                 all_hp_runs):
        self.model = model
        self.target = target
        self.training_error = training_error
        self.cv_error = cv_error
        self.holdout_error = holdout_error
        self.holdout_data = holdout_data
        self.metadata = metadata
        self.hyper_params = hyper_params
        self.all_hyper_param_runs = all_hp_runs

    def to_display_dict(self):
        return {
            'holdout error numeric': self.holdout_error,
            'holdout error': '{:.4g}'.format(self.holdout_error),
            'cv error': '{:.4g}'.format(self.cv_error),
            'training error': '{:.4g}'.format(self.training_error),
            'hyper_params': self.hyper_params
        }

