import numpy as np
from guacml.preprocessing.feature_generation.base_historic_target import BaseHistoricTarget


class LaggedTarget(BaseHistoricTarget):

    def __init__(self, lags, config, logger, group_keys=None):
        self.date_col = config['run_time']['time_series']['date_split_col']
        hist_param = lags
        super().__init__(hist_param, config, logger, group_keys)

    def get_hist_target_name(self):
        return 'lagged'

    def calc_hist_target(self, grouped_series, group_keys, hist_param):
        # nth = grouped_series.nth(hist_param)
        # # workaround because nth doesn't return the dates
        # cumcount = grouped_series.cumcount()
        # dates = pd.Series(cumcount[cumcount == hist_param].index,
        #                   index=nth.index,
        #                   name=self.date_col)
        # return pd.concat([dates, nth], axis=1).reset_index().set_index(self.date_col)

        shifted = grouped_series.shift(hist_param - 1).to_frame()
        # restore the group keys to the result
        for i, key in enumerate(group_keys):
            labels = grouped_series.grouper.labels[i]
            levels = grouped_series.grouper.levels[0]
            shifted[key] = np.array(levels)[labels]
        return shifted
