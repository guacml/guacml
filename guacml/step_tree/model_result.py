class ModelResult:

    def __init__(self,
                 model,
                 features,
                 target,
                 training_error,
                 cv_error,
                 holdout_error,
                 holdout_error_interval,
                 holdout_data,
                 metadata,
                 hyper_params,
                 display_hp_runs):
        self.model = model
        self.features = features
        self.target = target
        self.training_error = training_error
        self.cv_error = cv_error
        self.holdout_error = holdout_error
        self.holdout_error_interval = holdout_error_interval
        self.holdout_data = holdout_data
        self.metadata = metadata
        self.hyper_params = hyper_params
        self.display_hyper_param_runs = display_hp_runs

    def to_display_dict(self):
        if self.training_error is None:
            training_error_str = ''
        else:
            training_error_str = '{:.4g}'.format(self.training_error)

        return {
            'n features': len(self.features),
            'holdout error numeric': self.holdout_error,
            'holdout error': '{:.4g}'.format(self.holdout_error),
            'holdout error interval': '{:.4g}, {:.4g}'.format(self.holdout_error_interval[0],
                                                              self.holdout_error_interval[1]),
            'cv error': '{:.4g}'.format(self.cv_error),
            'training error': training_error_str,
            'hyper_params': self.hyper_params
        }



