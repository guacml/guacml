from guacml.util import deep_copy


class BaseStep:

    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.runtime = None
        self.state = None

    def copy(self, config):
        copy = self.__class__(config, self.logger)
        copy.state = deep_copy(self.state)

        return copy

    def execute(self, data):
        if self.config['run_time']['inplace'] is False:
            data = data.copy()

        self.execute_inplace(data)
        return data

    def execute_inplace(self, data):
        raise NotImplementedError()
