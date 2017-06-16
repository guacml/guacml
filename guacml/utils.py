import hyperopt.pyll.stochastic as py_st
import pandas as pd


def print_hp_distribution(dist, n_samples=200, bins='auto'):
    """
    This plot can help you find the right parameters for specifying
    priors for the hyper parameter optimization.
    """
    # dist = hp.qlognormal('n_estimators', 2, 1, 1)
    vals = pd.Series([py_st.sample(dist) for i in range(n_samples)])
    vals.hist(bins=bins)


def remove_outlier_rows(df, column, removal_ratio, side='both'):
    if side == 'both':
        lower = df[column].quantile(removal_ratio / 2)
        upper = df[column].quantile(1 - removal_ratio / 2)
        return df[df[column].between(lower, upper)]
    elif side == 'top':
        upper = df[column].quantile(1 - removal_ratio)
        return df[df.colum <= upper]
    elif side == 'bottom':
        lower = df[column].quantile(removal_ratio)
        return df[df.colum >= lower]
    else:
        raise ValueError('Unexpected value {} fof side parameter.'.format(side))
