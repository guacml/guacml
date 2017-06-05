import os
import yaml


def load_config():
    conf_path = os.path.join(os.path.dirname(__file__), 'test_config.yaml')
    with open(conf_path, 'r') as file:
        try:
            config = yaml.load(file)
        except yaml.YAMLError as exc:
            print(exc)

    return config
