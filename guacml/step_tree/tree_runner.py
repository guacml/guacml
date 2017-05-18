from guacml.step_tree.model_runner import ModelRunner


class TreeRunner:
    def __init__(self, dataset, tree):
        self.dataset = dataset
        self.tree = tree

    def run(self):
        result = {}
        self.run_step(self.tree.root_name, self.dataset.df, self.dataset.metadata, result)

        return result

    def run_step(self, step_name, input, metadata, accumulator):
        print('Running step ' + step_name)
        children = self.tree.get_children(step_name)
        step = self.tree.get_step(step_name)

        if isinstance(step, ModelRunner):
            step.splitter = self.dataset.splitter
            accumulator[step_name] = step.execute(input, metadata)
        else:
            output, metadata = step.execute(input, metadata)

            for child in children:
                self.run_step(child, output, metadata, accumulator)
