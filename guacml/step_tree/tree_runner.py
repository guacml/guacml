class TreeRunner:
    def __init__(self, dataset, tree):
        self.dataset = dataset
        self.tree = tree

    def run(self):
        result = {}
        self.run_step(self.tree.root_name, self.dataset.df, self.dataset.metadata, result)

        return result

    def run_step(self, step_name, input, metadata, accumulator):
        children = self.tree.get_children(step_name)
        step = self.tree.get_step(step_name)

        if not children:
            # Poor man's way to distinguish models from transformations - models are always leafs
            # `step` is an instance of ModelRunner here - need to fix that abstraction leak
            step.splitter = self.dataset.splitter
            accumulator[step_name] = step.execute(input, metadata)
        else:
            output, metadata = step.execute(input, metadata)

            for child in children:
                self.run_step(child, output, metadata, accumulator)
