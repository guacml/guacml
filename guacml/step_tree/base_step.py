class BaseStep:
    def __init__(self):
        self.runtime = None

    def execute(self, input, metadata):
        raise NotImplementedError()
