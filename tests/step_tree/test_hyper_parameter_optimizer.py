import unittest

from tests.test_util import load_dataset


class TestHyperParameterOptimizer(unittest.TestCase):

    def test_with_fixed_config(self):
        guac = load_dataset(config={
            'models': {'xgboost': {'hyper_parameters': {'n_rounds': 7, 'max_depth': 4}}}
        })
        guac.run(2)

        self.assertEquals(1, guac.model_results['xgboost'].display_hyper_param_runs.shape[0])
        self.assertEquals(7, guac.model_results['xgboost'].hyper_params['n_rounds'])
