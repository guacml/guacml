class HyperParameterInfo():

    def __init__(self, default, range, init_points):
        self.default = default
        self.range = range
        # has to be the same number for all hyper parameters of a model
        self.init_points = init_points
