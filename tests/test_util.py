import os
import yaml
import pandas as pd
from guacml import GuacMl


def load_dataset(fixture='titanic', target='Survived', eval_metric=None, config=None):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    df = pd.read_csv('{0}/fixtures/{1}.csv'.format(dir_path, fixture))
    guac = GuacMl(df, target, eval_metric=eval_metric, config=config)
    guac.clear_previous_runs()
    return guac


def load_config():
    conf_path = os.path.join(os.path.dirname(__file__), 'test_config.yaml')
    with open(conf_path, 'r') as file:
        try:
            config = yaml.load(file)
        except yaml.YAMLError as exc:
            print(exc)

    return config
