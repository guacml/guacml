from guacml.step_tree.model_manager import ModelManager
from guacml.storage.previous_runs import PreviousRuns
import time


class TreeRunner:
    def __init__(self, data, config, tree):
        self.data = data
        self.config = config
        self.tree = tree
        self.prev_runs = PreviousRuns(data, config)

    def run(self):
        """
        Either runs all preprocessing steps or loads the preprocessed
        data from files, if they have been run before.
        """
        if not self.prev_runs.found_matching_run:
            results = {}
            self.run_step(self.tree.root_name, self.data, results)
            self.prev_runs.store_run()
            return results
        else:
            return self.load_previous()

    def run_step(self, step_name, data, results):
        print('Running step ' + step_name)
        children = self.tree.get_children(step_name)
        step = self.tree.get_step(step_name)

        if isinstance(step, ModelManager):
            results[step_name] = self.execute_step_with_timing(step, data)

            self.prev_runs.add_model_result(step_name, results[step_name])
        else:
            dataset = self.execute_step_with_timing(step, data)
            for child in children:
                self.run_step(child, dataset, results)

    def load_previous(self):
        return self.prev_runs.get_prev_results()

    def clear_prev_runs(self):
        self.prev_runs.clear()

    @staticmethod
    def execute_step_with_timing(step, data):
        start = time.perf_counter()
        result = step.execute(data)
        end = time.perf_counter()
        step.runtime = end - start

        return result


