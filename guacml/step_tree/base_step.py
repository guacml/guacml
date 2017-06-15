class BaseStep:
    def __init__(self):
        self.runtime = None

    def execute(self, data):
        raise NotImplementedError()
