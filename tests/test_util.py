import os
import yaml
import pandas as pd
from guacml import GuacMl
from guacml.util import deep_update


def read_fixture(fixture='titanic'):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    return pd.read_csv('{0}/fixtures/{1}.csv'.format(dir_path, fixture))


def load_dataset(fixture='titanic', target='Survived', eval_metric=None,
                 target_transform=None, config=None):
    df = read_fixture(fixture)
    default_config = load_config()

    if config is not None:
        deep_update(default_config, config)

    return GuacMl(df, target, eval_metric=eval_metric,
                  target_transform=target_transform, config=default_config)


def load_config():
    conf_path = os.path.join(os.path.dirname(__file__), 'test_config.yaml')
    with open(conf_path, 'r') as file:
        try:
            config = yaml.load(file)
        except yaml.YAMLError as exc:
            print(exc)

    return config
