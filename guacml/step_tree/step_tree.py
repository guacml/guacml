import pydot

from guacml.models.base_model import BaseModel
from .base_step import BaseStep
from collections import defaultdict
from .model_runner import ModelRunner


class StepTree:
    def __init__(self, target, hyper_param_iterations, eval_metric):
        self.steps = {}
        self.children = defaultdict(list)
        self.target = target
        self.hyper_param_iterations = hyper_param_iterations
        self.eval_metric = eval_metric
        self.root_name = None

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
        self.add_step(step_name, parent_name, ModelRunner(model,
                                                          self.target,
                                                          self.hyper_param_iterations,
                                                          self.eval_metric))

    def get_step(self, name):
        return self.steps[name]

    def get_children(self, step_name):
        return self.children[step_name]

    def to_pydot(self):
        """Return a pydot digraph from a StepTree."""

        graph = pydot.Dot(graph_type='digraph')

        # make sure Root shows up in single node trees
        if self.root_name:
            graph.add_node(pydot.Node(self.root_name))

        for parent, children in self.children.items():
            for child in children:
                graph.add_edge(pydot.Edge(parent, child))

        return graph
