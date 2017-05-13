class ModelRunner:
    def __init__(self, step, splitter):
        self.step = step
        self.splitter = splitter

    def run(self, input, metadata):
        train, cv = self.splitter.split(input)
        # TODO do something useful with CV
        return self.step.execute(train, metadata)
