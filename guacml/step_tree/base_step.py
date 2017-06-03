class BaseStep:
    def execute(self, dataframe, metadata):
        raise NotImplementedError()
