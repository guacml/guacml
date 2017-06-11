from guacml.models.base_model import BaseModel
from .base_step import BaseStep
from collections import defaultdict
from .model_manager import ModelManager


class StepTree:
    def __init__(self, config):
        self.steps = {}
        self.children = defaultdict(list)
        self.config = config

    def add_step(self, step_name, parent_name, step):
        if parent_name is None:
            if len(self.steps) > 0:
                raise Exception('Parent can only be empty for the first node added.')
            self.root_name = step_name
        elif parent_name not in self.steps:
            raise ValueError('Parent {0} not present.'.format(parent_name))
        if step_name in self.steps:
            raise ValueError('Step {0} alrady present.'.format(step_name))
        if not isinstance(step, BaseStep):
            raise ValueError('Argument step needs to be derived from BaseStep.')

        self.steps[step_name] = step
        if not parent_name is None:
            self.children[parent_name].append(step_name)

    def add_model(self, step_name, parent_name, model):
        if not isinstance(model, BaseModel):
            raise ValueError('The model paramter should inherit from BaseModel')
        self.add_step(step_name, parent_name, ModelManager(model, self.config))

    def get_step(self, name):
        return self.steps[name]

    def get_children(self, step_name):
        return self.children[step_name]
