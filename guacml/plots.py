import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from guacml.preprocessing.column_analyzer import ColType


class Plots:
    def __init__(self, model_results):
        self.model_results = model_results

    def model_overview(self):
        rows = []
        for name, res in self.model_results.items():
            res_dict = res.to_display_dict()
            res_dict['model name'] = name
            rows.append(res_dict)
        result = pd.DataFrame(rows, columns=['model name', 'holdout error', 'holdout accuracy', 'cv error', 'training error'])
        return result.sort_values('holdout error')

    def error_overview(self, bins='auto', figsize=(8, 6)):
        n_models = len(self.model_results)
        fig, axes = plt.subplots(n_models, sharex=True, sharey=True)
        fig.subplots_adjust(hspace=0.2)
        for idx, value in enumerate(self.model_results.items()):
            name, model = value
            model.holdout_data['error'].hist(bins=bins, figsize=figsize, ax=axes[idx], label=name)
            axes[idx].legend()
            axes[idx].set_ylabel('n rows')
        plt.xlabel('error')
        plt.suptitle('Model error histograms', fontsize=14)
        plt.show()

    def model_error_by_feature(self, model_name, data, metadata):
        model_results = self.model_results[model_name]
        data['error'] = model_results.holdout_data['error']
        holdout = data[data.error.notnull()]

        low_card_cols = [ColType.CATEGORICAL, ColType.ORDINAL]
        for col in metadata[metadata.type.isin(low_card_cols)].col_name:
            sns.barplot(x=col, y='error', data=holdout)
            plt.show()

    def predictions_vs_actual(self, model_name, n_bins = 5, figsize=(7, 5)):
        model_resuls = self.model_results[model_name]
        holdout = model_resuls.holdout_data
        target = model_resuls.target
        bins = np.arange(0, 1.001, 1 / n_bins)
        binned = pd.cut(holdout['prediction'], bins=bins)
        bin_counts = holdout.groupby(binned)[target].count()
        bin_means = holdout.groupby(binned)[target].mean()

        fig, axes = plt.subplots(2, sharex=True, figsize=figsize)
        axes[0].bar(bins[:-1], bin_means, width=1/n_bins)
        axes[0].plot(bins, bins, color=sns.color_palette()[1])
        axes[0].set_ylabel('actual rate')
        axes[1].bar(bins[:-1], bin_counts, width=1/n_bins)
        axes[1].set_ylabel('number of rows')
        plt.xlabel('predicted probability')
        plt.suptitle('Predictions vs Actual', fontsize=14)





