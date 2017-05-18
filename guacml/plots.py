import matplotlib.pyplot as plt
import seaborn as sns

from guacml.preprocessing.column_analyzer import ColType


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

    def model_error_by_feature(self, model_name, data, metadata):
        model_results = self.model_results[model_name]
        data['error'] = model_results.holdout_row_errors
        holdout = data[data.error.notnull()]

        low_card_cols = [ColType.CATEGORICAL, ColType.ORDINAL]
        for col in metadata[metadata.type.isin(low_card_cols)].col_name:
            sns.barplot(x=col, y='error', data=holdout)
            plt.show()



