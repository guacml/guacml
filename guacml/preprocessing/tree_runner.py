class TreeRunner:
    def __init__(self, tree):
        self.tree = tree

    def run(self, input):
        result = {}
        self.run_step(self.tree.root_name, input, result)

        return result

    def run_step(self, step_name, input, accumulator):
        output = self.tree.get_step(step_name).execute(input)
        children = self.tree.get_children(step_name)

        if not children:
            accumulator[step_name] = output
        else:
            for child in children:
                self.run_step(child, output, accumulator)
