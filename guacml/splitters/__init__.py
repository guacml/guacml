from guacml.splitters.random_splitter import RandomSplitter
from guacml.splitters.date_splitter import DateSplitter


def create(config):
    if config['run_time']['time_series'].get('date_split_col') is not None:
        return DateSplitter(config)
    else:
        return RandomSplitter(config['cross_validation'])
