import base64

from guacml.dataset import Dataset
from guacml.preprocessing.column_analyzer import ColumnAnalyzer
from guacml.step_tree.step_tree import StepTree
from guacml.util import deep_copy, get_fully_qualified_class_name, get_class_from_string


class Pipeline:

    @staticmethod
    def from_tree(guac, model_name):
        mr = guac.model_results[model_name]
        tree = guac.tree.get_subtree_upto(model_name)
        config = tree.config
        config['column_types'] = mr.metadata.type.to_dict()
        logger = guac.logger
        model = mr.model.copy(config)
        features = list(mr.features)

        return Pipeline(model_name, tree, config, logger, model, features)

    def __init__(self, name, tree, config, logger, model, features):
        self.name = name
        self.tree = tree
        self.config = config
        self.logger = logger
        self.model = model
        self.features = features

    def transform(self, test_set):
        step_name = self.tree.root_name
        metadata = ColumnAnalyzer(self.config, self.logger).analyze(test_set)
        data = Dataset(test_set, metadata)

        while step_name is not None:
            step = self.tree.get_step(step_name)
            data = step.execute(data)
            children = self.tree.get_children(step_name)
            step_name = children[0] if len(children) > 0 else None

        return data.df

    def predict(self, test_set):
        df = self.transform(test_set)

        return self.model.predict(df[self.features])

    def serialize(self):
        def get_step_state(step): return {'state': deep_copy(step.state), 'class': get_fully_qualified_class_name(step)}
        tree = {
            'steps': {step_name: get_step_state(step) for step_name, step in self.tree.steps.items()},
            'children': dict(self.tree.children),
        }
        model = {
            'class': get_fully_qualified_class_name(self.model),
            'state': base64.b64encode(self.model.get_state()).decode(),
        }

        return {
            'name': self.name,
            'config': deep_copy(self.config),
            'features': list(self.features),
            'tree': tree,
            'model': model,
        }

    def deserialize(state, logger):
        name = state['name']
        config = state['config']
        tree = deserialize_tree(state['tree'], config, logger)
        model = deserialize_model(state['model'], config, logger)
        features = state['features']

        return Pipeline(name, tree, config, logger, model, features)


def deserialize_tree(tree_state, config, logger):
    tree = StepTree(config, logger)
    steps = tree_state['steps']
    children = tree_state['children']
    step_names = set(steps)
    all_children = set(sum(children.values(), []))
    candidates_for_root = step_names - all_children

    if len(candidates_for_root) != 1:
        raise Exception('Unable to determine root from {}'.format(tree_state))

    parent = candidates_for_root.pop()

    tree.add_step(parent, None, deserialize_step(steps[parent], config, logger))

    while parent in children and len(children[parent]) > 0:
        cs = children[parent]

        if len(cs) > 1:
            raise Exception('Each step should have one child, but {} has {}'.format(parent, cs))

        child = cs[0]
        tree.add_step(child, parent, deserialize_step(steps[child], config, logger))
        parent = child

    return tree


def deserialize_step(step_state, config, logger):
    step_class = get_class_from_string(step_state['class'])
    step = step_class(config, logger)
    step.state = step_state['state']

    return step


def deserialize_model(model_state, config, logger):
    model_class = get_class_from_string(model_state['class'])
    model = model_class(config, logger)
    model.set_state(base64.b64decode(model_state['state'].encode()))

    return model
