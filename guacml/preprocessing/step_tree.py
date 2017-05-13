from .base_step import BaseStep
from collections import defaultdict


class StepTree:
    def __init__(self):
        self.steps = {}
        self.children = defaultdict(list)
        self.parents = {}

    def add_step(self, step_name, parent_name, step):
        if parent_name is None and len(self.steps) > 0:
            raise Exception('Parent can only be empty for the first node added.')
        if step_name in self.steps:
            raise ValueError('Step {0} alrady present.'.format(step_name))
        if parent_name not in self.steps:
            raise ValueError('Parent {0} not present.'.format(parent_name))
        if not isinstance(step, BaseStep):
            raise ValueError('Argument step needs to be derived from BaseFeatureGenerator.')

        self.steps[step_name] = step
        self.parents[step_name] = parent_name
        if not parent_name is None:
            self.children[parent_name].append(step_name)

    def get_children(self, step_name):
        return self.children[step_name]

    def path_from_root(self, step_name):
        path = [step_name]
        parent = self.parents[step_name]
        while not parent is None:
            path.append(parent)
            parent = self.parents[parent]
        return path.reverse()