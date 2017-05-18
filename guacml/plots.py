import matplotlib.pyplot as plt


class Plots:
    def __init__(self, model_results):
        self.model_results = model_results

    def error_overview(self, bins='auto', figsize=(8, 6)):
        n_models = len(self.model_results)
        fig, axes = plt.subplots(n_models, sharex=True, sharey=True)
        fig.subplots_adjust(hspace=0.2)
        for idx, value in enumerate(self.model_results.items()):
            name, model = value
            model.holdout_row_errors.hist(bins=bins, figsize=figsize, ax=axes[idx], label=name)
            axes[idx].legend()
            axes[idx].set_ylabel('n rows')
        plt.xlabel('error')
        plt.suptitle('Model error histograms', fontsize=16)
        plt.show()

    def model_error_by_feature(self, model_name):
        model_results = self.model_results[model_name]
        data = model_results.holdout_data
        all_features = data.columns
        data['error'] = model_results.holdout_row_errors


