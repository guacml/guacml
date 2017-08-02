import base64

from guacml.dataset import Dataset
from guacml.preprocessing.column_analyzer import ColumnAnalyzer
from guacml.util import deep_copy, get_fully_qualified_class_name


class Pipeline:

    def __init__(self, guac, model_name):
        mr = guac.model_results[model_name]
        self.name = model_name
        self.tree = guac.tree.get_subtree_upto(model_name)
        self.config = self.tree.config
        self.config['column_types'] = mr.metadata.type.to_dict()
        self.logger = guac.logger
        self.model = mr.model.copy(self.config)
        self.features = list(mr.features)

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
