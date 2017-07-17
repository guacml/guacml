import pandas as pd


def analyze_frequency(df, ts_config):
    ts_date_col = ts_config['date_split_col']
    ts_key_cols = ts_config['series_key_cols']
    return analyze_frequency_for_group(df, ts_date_col, ts_key_cols)


def analyze_frequency_for_group(df, date_col, group_cols):
    dates_and_keys = df[[date_col] + group_cols].sort_values(date_col)
    shifted = dates_and_keys.groupby(group_cols)[date_col].shift(1)

    diffs = (dates_and_keys[date_col] - shifted)
    diff_value_counts = diffs.value_counts()
    frequency = diff_value_counts.index[0]
    for diff, count in diff_value_counts.iteritems():
        if frequency == pd.Timedelta(0):
            raise ValueError('Many duplicate dates found in time series. If these dates belong to '
                             'different series, specify the key for the series in'
                             'make_time_series with the parameter series_key_col.')
        if diff % frequency != pd.Timedelta(0):
            raise ValueError('Can not determine frequency of time series. Found gap of length {}, '
                             'which is not a multiple of the assumed frequency of {}'
                             .format(diff, frequency))
    return frequency
