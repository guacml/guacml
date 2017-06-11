from guacml.step_tree.model_manager import ModelManager


class TreeRunner:
    def __init__(self, data, tree):
        self.data = data
        self.tree = tree

    def run(self):
        result = {}
        self.run_step(self.tree.root_name, self.data, result)

        return result

    def run_step(self, step_name, data, accumulator):
        print('Running step ' + step_name)
        children = self.tree.get_children(step_name)
        step = self.tree.get_step(step_name)

        if isinstance(step, ModelManager):
            accumulator[step_name] = step.execute(data)
        else:
            dataset = step.execute(data)

            for child in children:
                self.run_step(child, dataset, accumulator)
