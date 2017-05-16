import hyperopt.pyll.stochastic as py_st
import pandas as pd

def print_hp_distribution(dist, n_samples=200, bins='auto'):
    # dist = hp.qlognormal('n_estimators', 2, 1, 1)
    vals = pd.Series([py_st.sample(dist) for i in range(n_samples)])
    vals.hist(bins=bins)