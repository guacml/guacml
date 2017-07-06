import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from guacml.enums import ProblemType
from guacml.preprocessing.column_analyzer import ColType
import guacml.utils as utils


class Plots:
    def __init__(self, run_time_config, input_data, step_tree):
        self.model_results = None
        self.problem_type = run_time_config['problem_type']
        self.target = run_time_config['target']
        self.run_time_config = run_time_config
        self.input_data = input_data
        self.step_tree = step_tree

    def set_model_results(self, model_results):
        self.model_results = model_results

    def target_plot(self):
        target_type = self.input_data.metadata.loc[self.target].type
        target_data = self.input_data.df[self.target]
        sns.set(style="white", color_codes=True)
        if not self.run_time_config['is_time_series']:
            if target_type == ColType.BINARY:
                plt.figure(figsize=(6, 1))
                sns.barplot(target_data.sum() / target_data.shape[0])
                plt.xlim([0, 1])
                plt.title(target_data.name + ' rate')
            elif target_type == ColType.NUMERIC or target_type == ColType.ORDINAL:
                plt.figure(figsize=(6, 2))
                ax = sns.distplot(target_data, hist_kws=dict(edgecolor='black'))
                ax.set_xlim(target_data.min(), target_data.max())
                plt.title(target_data.name + ' histogram')
        else:
            self.time_series_target_plot()

    def time_series_target_plot(self):
        date_col = self.run_time_config['time_series']['date_split_col']
        series_key_cols = self.run_time_config['time_series']['series_key_cols']
        df = self.input_data.df

        keys_sample = self._get_time_series_key_sample()

        n_key_combs = keys_sample.shape[0]
        fig, axes = plt.subplots(n_key_combs, 2, figsize=(12, 3 * n_key_combs),
                     gridspec_kw={'width_ratios':[5, 2]})

        for i, pair in enumerate(keys_sample.iterrows()):
            if len(axes.shape) == 1:
                ax_row = axes
            else:
                ax_row = axes[i]

            keys = pair[1]
            time_series = df[(df[series_key_cols] == keys).all(axis=1)]
            time_series = time_series.set_index(date_col).sort_index()
            time_series[self.target].plot(ax=ax_row[0])
            ax_row[0].set_title(dict(keys))
            short_range_min = -5 * self.run_time_config['time_series']['prediction_length']
            last_dates = time_series.index[short_range_min:]
            time_series.loc[last_dates, self.target].plot(ax=ax_row[1])

        fig.tight_layout()

    def _get_time_series_key_sample(self):
        if not hasattr(self, 'ts_key_sample'):
            series_key_cols = self.run_time_config['time_series']['series_key_cols']
            key_combinations = self.input_data.df[series_key_cols].drop_duplicates()
            self.ts_key_sample = key_combinations.sample(min(5, key_combinations.shape[0]))
        return self.ts_key_sample

    def error_overview(self, bins='auto', figsize=(8, 6)):
        n_models = len(self.model_results)
        fig, axes = plt.subplots(n_models, sharex=True, sharey=True)
        fig.subplots_adjust(hspace=0.2)
        for idx, value in enumerate(self.model_results.items()):
            name, model = value
            model.holdout_data['error'].hist(bins=bins, figsize=figsize, ax=axes[idx], label=name,
                                             edgecolor='black')
            axes[idx].legend()
            axes[idx].set_ylabel('n rows')
            x_max = axes[idx].get_xlim()[1]
            axes[idx].set_xlim(model.holdout_data['error'].min(), x_max)
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

    def tree(self):
        # render pydot by calling dot, no file saved to disk
        png_str = self.step_tree.to_pydot().create_png(prog='dot')

        from IPython.core.display import Image
        return Image(png_str, embed=True)

    def predictions_vs_actual(self, model_name, n_bins=10, **kwargs):
        model_result = self.model_results[model_name]
        if not self.run_time_config['is_time_series']:
            if self.problem_type == ProblemType.BINARY_CLAS:
                return predictions_vs_actual_classification(model_result, model_name, n_bins, **kwargs)
            elif self.problem_type == ProblemType.REGRESSION:
                return predictions_vs_actual_regression(model_result, model_name, **kwargs)
            else:
                raise Exception('Not implemented for problem type ' + self.problem_type)
        else:
            self.predictions_vs_actual_time_series(model_result, model_name, self.run_time_config, **kwargs)

    def predictions_vs_actual_time_series(self, model_results, model_name,  size=6, **kwargs):
        date_col = self.run_time_config['time_series']['date_split_col']
        series_key_cols = self.run_time_config['time_series']['series_key_cols']
        keys_sample = self._get_time_series_key_sample()

        holdout = model_results.holdout_data
        target = model_results.target

        n_key_combs = keys_sample.shape[0]
        fig, axes = plt.subplots(n_key_combs, 1, figsize=(12, 3 * n_key_combs))

        for i, pair in enumerate(keys_sample.iterrows()):
            if isinstance(axes, np.ndarray):
                ax= axes[i]
            else:
                ax= axes

            keys = pair[1]
            time_series = holdout[(holdout[series_key_cols] == keys).all(axis=1)]
            time_series = time_series.set_index(date_col)
            time_series['prediction'].plot(ax=ax, figsize=(12, 3 * n_key_combs))
            time_series[self.target].plot(ax=ax)
            ax.set_title(dict(keys))

        fig.tight_layout()


def predictions_vs_actual_classification(model_results, model_name, n_bins, figsize=(7, 3)):
    holdout = model_results.holdout_data
    target = model_results.target
    bins = np.arange(0, 1.001, 1 / n_bins)
    bin_mids = (bins[:-1] + bins[1:]) / 2
    binned = pd.cut(holdout['prediction'], bins=bins)
    bin_counts = holdout.groupby(binned)[target].count()
    bin_means = holdout.groupby(binned)[target].mean()

    fig = plt.figure(figsize=figsize)
    plt.suptitle('{0}: Predictions vs Actual'.format(model_name), fontsize=14)
    ax1 = plt.gca()
    ax1.grid(False)
    ax1.bar(bin_mids, bin_counts, width=1/n_bins, color=sns.light_palette('green')[1],
            label='row count', edgecolor='black')
    ax1.set_xlabel('predicted probability')
    ax1.set_ylabel('row count')

    ax2 = ax1.twinx()
    ax2.plot(bin_mids, bin_means, linewidth=3,
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
    return fig


def predictions_vs_actual_regression(model_results, model_name, size=6, bins=None,
                                     gridsize=30, outlier_ratio=None, **kwargs):
    holdout = model_results.holdout_data
    target = model_results.target

    if outlier_ratio is not None:
        holdout = utils.remove_outlier_rows(holdout, 'prediction', outlier_ratio)
        holdout = utils.remove_outlier_rows(holdout, target, outlier_ratio)

    sns.set(style="white", color_codes=True)

    marginal_kws = dict(hist_kws=dict(edgecolor='black'))
    plt.suptitle('{0}: Predictions vs Actual'.format(model_name), fontsize=14)
    grid = sns.jointplot('prediction', target, holdout, 'hexbin', gridsize=gridsize,
                         size=size, bins=bins, space=0, marginal_kws=marginal_kws, **kwargs)
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)  # shrink fig so cbar is visible
    cax = grid.fig.add_axes([.95, .18, .04, .5])  # x, y, width, height
    color_bar = sns.plt.colorbar(cax=cax)

    if bins is None:
        color_bar.set_label('count')
    elif bins == 'log':
        color_bar.set_label('log_10(count)')
    return grid





