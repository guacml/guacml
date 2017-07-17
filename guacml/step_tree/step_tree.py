import pydot

from guacml.models.base_model import BaseModel
from guacml.step_tree.base_step import BaseStep
from collections import defaultdict
from guacml.step_tree.model_manager import ModelManager


class StepTree:
    def __init__(self, config, logger):
        self.steps = {}
        self.parents = {}
        self.children = defaultdict(list)
        self.config = config
        self.root_name = None
        self.logger = logger

    def add_step(self, step_name, parent_name, step):
        if step_name in self.steps:
            raise ValueError('Step {} alrady present.'.format(step_name))
        if parent_name is None:
            if len(self.steps) > 0:
                raise Exception('Parent can only be empty for the first node added.')
            self.root_name = step_name
        elif parent_name not in self.steps:
            raise ValueError('Parent {} not present.'.format(parent_name))
        if not (isinstance(step, BaseStep) or isinstance(step, ModelManager)):
            raise ValueError('Argument step of type {} needs to be derived from'
                             ' BaseStep or ModelManager.'.format(type(step)))
        self.steps[step_name] = step
        if parent_name is not None:
            self.add_parent_child_relation(parent_name, step_name)

    def add_parent_child_relation(self, parent, child):
        self.children[parent].append(child)
        self.parents[child] = parent

    def add_model(self, step_name, parent_name, model):
        if not isinstance(model, BaseModel):
            raise ValueError('The model parameter should inherit from BaseModel')
        self.add_step(step_name, parent_name, ModelManager(model, self.config, self.logger))

    def insert_step_before(self, step_name, insert_point, step):
        if step_name in self.steps:
            raise ValueError('Step {} alrady present.'.format(step_name))
        if insert_point not in self.steps:
            raise ValueError('Insert point {} not present.'.format(insert_point))
        if not isinstance(step, BaseStep):
            raise ValueError('Argument step of type {} needs to be derived from BaseStep.'
                             .format(type(step)))

        if insert_point == self.root_name:
            self.root_name = step_name
        else:
            parent = self.parents[insert_point]
            self.delete_parent_child(parent, insert_point)
            self.add_parent_child_relation(parent, step_name)

        self.add_parent_child_relation(step_name, insert_point)
        self.steps[step_name] = step

    def insert_step_after(self, step_name, insert_point, step):
        if step_name in self.steps:
            raise ValueError('Step {} alrady present.'.format(step_name))
        if insert_point not in self.steps:
            raise ValueError('Insert point {} not present.'.format(insert_point))
        if not isinstance(step, BaseStep):
            raise ValueError('Argument step of type {} needs to be derived from BaseStep.'
                             .format(type(step)))

        children = self.children[insert_point].copy()
        for child in children:
            self.delete_parent_child(insert_point, child)
            self.add_parent_child_relation(step_name, child)

        self.add_parent_child_relation(insert_point, step_name)
        self.steps[step_name] = step

    def delete_parent_child(self, parent, child):
        children = self.children[parent]
        if len(children) > 1:
            self.children[parent].remove(child)
        elif len(children) == 1:
            del self.children[parent]
        else:
            raise Exception('There should never be a children entry without items.')
        del self.parents[child]

    def get_step(self, name):
        return self.steps[name]

    def get_children(self, step_name):
        return self.children[step_name]

    def get_leaf_names(self):
        leaf_names = []
        self.get_children_or_leaf_(self.root_name, leaf_names)
        return leaf_names

    def get_children_or_leaf_(self, step_name, leaf_names):
        children = self.get_children(step_name)
        if len(children) == 0:
            leaf_names.append(step_name)

        for child_step_name in children:
            self.get_children_or_leaf_(child_step_name, leaf_names)

        return leaf_names

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
