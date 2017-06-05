from guacml.step_tree.model_manager import ModelManager
import pandas as pd


# ToDo: Refactor the file handling into separate class
from guacml.storage.previous_runs import PreviousRuns


class TreeRunner:
    def __init__(self, data, config, tree):
        self.data = data
        self.config = config
        self.tree = tree
        self.min_hp_iterations = None
        self.prev_runs = PreviousRuns(data, config)


    def run(self, min_hyper_param_iterations):
        """
        Either runs all preprocessing steps or loads the preprocessed
        data from files, if they have been run before
        """
        self.min_hp_iterations = min_hyper_param_iterations

        if not self.prev_runs.exists:
            results = {}
            self.run_step(self.tree.root_name, self.data, results)
            self.prev_runs.store_run()
            return results
        else:
            model_step_names = self.tree.get_leaf_names()
            for model_name in model_step_names:
                data = self.prev_runs.get_model_input(model_name)
                model_step = self.tree.get_step(model_name)
                model_step.execute(data, self.min_hp_iterations)

    def run_step(self, step_name, data, results):
        print('Running step ' + step_name)
        children = self.tree.get_children(step_name)
        step = self.tree.get_step(step_name)

        if isinstance(step, ModelManager):
            results[step_name] = step.execute(data, self.min_hp_iterations)
            self.prev_runs.add_model_input(step_name, data)
        else:
            dataset = step.execute(data)
            for child in children:
                self.run_step(child, dataset, results)
