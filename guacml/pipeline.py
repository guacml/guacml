from guacml.dataset import Dataset
from guacml.preprocessing.column_analyzer import ColumnAnalyzer


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
