class BaseStep:

    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.runtime = None
        self.state = None

    def execute(self, data):
        if self.config['run_time']['inplace'] is False:
            data = data.copy()

        self.execute_inplace(data)
        return data

    def execute_inplace(self, data):
        raise NotImplementedError()
