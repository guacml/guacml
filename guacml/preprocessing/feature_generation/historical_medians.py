from guacml.preprocessing.feature_generation.base_historic_target import BaseHistoricTarget


class HistoricalMedians(BaseHistoricTarget):

    def __init__(self, hist_length_factors, config, logger, group_keys=None):
        prediction_length = config['run_time']['time_series']['prediction_length']
        median_windows = [length_factor * prediction_length
                          for length_factor in hist_length_factors]
        hist_parameters = median_windows
        super().__init__(hist_parameters, config, logger, group_keys)

    def get_hist_target_name(self):
        return 'median'

    def calc_hist_target(self, grouped_series, group_keys, hist_param):
        return grouped_series.rolling(hist_param, min_periods=1) \
                             .median() \
                             .reset_index(group_keys)
