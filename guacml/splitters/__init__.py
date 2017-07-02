from guacml.splitters.random_splitter import RandomSplitter
from guacml.splitters.date_splitter import DateSplitter


def create(config):
    if 'date_split_col' not in config['run_time']['time_series']:
        return RandomSplitter(config['cross_validation'])
    else:
        return DateSplitter(config)
