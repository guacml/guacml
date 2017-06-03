import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
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
            model.holdout_data['error'].hist(bins=bins, figsize=figsize, ax=axes[idx], label=name)
            axes[idx].legend()
            axes[idx].set_ylabel('n rows')
        plt.xlabel('error')
        plt.suptitle('Model error histograms', fontsize=14)
        plt.show()

    def model_error_by_feature(self, model_name):
        model_results = self.model_results[model_name]
        metadata = model_results.metadata
        holdout = model_results.holdout_data
        low_card_cols = [ColType.CATEGORICAL, ColType.ORDINAL]
        for col in metadata[metadata.type.isin(low_card_cols)].col_name:
            sns.barplot(x=col, y='error', data=holdout)
            plt.show()

    def predictions_vs_actual(self, model_name, n_bins = 10, figsize=(7, 4)):
        model_resuls = self.model_results[model_name]
        holdout = model_resuls.holdout_data
        target = model_resuls.target
        bins = np.arange(0, 1.001, 1 / n_bins)
        binned = pd.cut(holdout['prediction'], bins=bins)
        bin_counts = holdout.groupby(binned)[target].count()
        bin_means = holdout.groupby(binned)[target].mean()

        plt.figure(figsize=figsize)
        plt.title('{0}: Predictions vs Actual'.format(model_name), fontsize=14)
        ax1 = plt.gca()
        ax1.grid(False)
        ax1.bar(bins[:-1], bin_counts, width=1/n_bins, color=sns.light_palette('green')[1],
                label='row count', linewidth = 1, edgecolor='grey')
        ax1.set_xlabel('predicted probability')
        ax1.set_ylabel('row count')

        ax2 = ax1.twinx()
        ax2.plot((bins[:-1]+bins[1:]) / 2 , bin_means, linewidth=3,
                 marker='.', markersize=16, label='actual rate')
        ax2.plot(bins, bins, color=sns.color_palette()[2], label='main diagonal')

        ax2.set_ylabel('actual rate')

        handles, labels = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        legend = plt.legend(handles + handles2, labels + labels2,
                   loc='best',
                   frameon=True,
                   framealpha=0.7)
        frame = legend.get_frame()
        frame.set_facecolor('white')
        plt.show()






