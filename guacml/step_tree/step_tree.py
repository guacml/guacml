import pydot

from guacml.models.base_model import BaseModel
from .base_step import BaseStep
from collections import defaultdict
from .model_manager import ModelManager


class StepTree:
    def __init__(self, config):
        self.steps = {}
        self.children = defaultdict(list)
        self.config = config
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
        if not (isinstance(step, BaseStep) or isinstance(step, ModelManager)):
            raise ValueError('Argument step of type {0} needs to be derived from BaseStep.'
                             .format(type(step)))
        self.steps[step_name] = step
        if parent_name is not None:
            self.children[parent_name].append(step_name)

    def add_model(self, step_name, parent_name, model):
        if not isinstance(model, BaseModel):
            raise ValueError('The model parameter should inherit from BaseModel')
        self.add_step(step_name, parent_name, ModelManager(model, self.config))

    def get_step(self, name):
        return self.steps[name]

    def get_children(self, step_name):
        return self.children[step_name]

    def get_leaf_names(self):
        leaf_names = []
        self.get_children_or_leaf_(self.root_name, leaf_names)
        return leaf_names

    def get_children_or_leaf_(self, step_name, leaf_names):
        for child_step_name in self.get_children(step_name):
            child_step = self.steps[child_step_name]
            if isinstance(child_step, BaseModel):
                leaf_names.append(child_step_name)
            else:
                self.get_children_or_leaf_(step_name, leaf_names)

    def to_pydot(self):
        """Return a pydot digraph from a StepTree."""

        graph = pydot.Dot(graph_type='digraph')

        # make sure Root shows up in single node trees
        if self.root_name:
            graph.add_node(pydot.Node(self._step_graph_label(self.root_name)))

        for parent, children in self.children.items():
            for child in children:
                graph.add_edge(pydot.Edge(
                    self._step_graph_label(parent),
                    self._step_graph_label(child)
                    ))

        return graph

    def _step_graph_label(self, step_name):
        step = self.get_step(step_name)
        if hasattr(step, 'runtime') and step.runtime is not None:
            return "%s\n(%.2f sec)" % (step_name, step.runtime)
        else:
            return step_name
