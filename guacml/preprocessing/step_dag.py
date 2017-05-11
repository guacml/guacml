from .feature_generation import BaseFeatureGenerator

class StepDag:
    def __init__(self):
        self.steps = {}
        self.edges = set()

    def add_step(self, name, step):
        if name in self.steps:
            raise ValueError('Step alrady present')
        if not isinstance(step, BaseFeatureGenerator):
            raise ValueError('Argument step needs to be derived from BaseFeatureGenerator')
        self.steps[name] = step

    def add_edge(self, earlier, later):
        if not earlier in self.steps:
            raise ValueError('Earlier step not known.')
        if not later in self.steps:
            raise ValueError('Later step not known.')
        edge = (earlier, later)
        if edge in self.edges:
            raise ValueError('Edge alrady present')
        self.edges.add(edge)