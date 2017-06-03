from guacml.splitters.random_splitter import RandomSplitter
from guacml.splitters.date_splitter import DateSplitter


def create(config):
    if config['run_time']['date_split_col'] is None:
        return RandomSplitter(config['cross_validation'])
    else:
        return DateSplitter(config)
