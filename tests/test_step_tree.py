import unittest

from guacml.step_tree.step_tree import StepTree
from guacml.step_tree.base_step import BaseStep


class TestStepTree(unittest.TestCase):
    def _new_empty_tree(self):
        empty_conf = {}
        return StepTree(empty_conf)

    def _step_with_runtime(self, runtime):
        step = BaseStep()
        step.runtime = runtime
        return step

    def test_to_pydot(self):
        # empty tree gives empty result
        step_tree = self._new_empty_tree()
        self.assertEqual("digraph G {\n}\n", step_tree.to_pydot().to_string())

        # one node tree
        step_tree.add_step("Root", None, BaseStep())
        self.assertEqual("digraph G {\nRoot;\n}\n", step_tree.to_pydot().to_string())

        # one child
        step_tree.add_step("FirstChild", "Root", BaseStep())
        self.assertEqual("digraph G {\nRoot;\nRoot -> FirstChild;\n}\n",
                         step_tree.to_pydot().to_string())

        # child with childs
        step_tree.add_step("SecondChild", "Root", BaseStep())
        step_tree.add_step("FirstGrandChild", "FirstChild", BaseStep())
        step_tree.add_step("SecondGrandChild", "FirstChild", BaseStep())
        self.assertEqual((
            "digraph G {\n"
            "Root;\n"
            "Root -> FirstChild;\n"
            "Root -> SecondChild;\n"
            "FirstChild -> FirstGrandChild;\n"
            "FirstChild -> SecondGrandChild;\n"
            "}\n"
            ), step_tree.to_pydot().to_string())

    def test_to_pydot_with_runtime(self):
        step_tree = self._new_empty_tree()

        # one node tree
        step_tree.add_step("Root", None, self._step_with_runtime(0.1))
        self.assertEqual('digraph G {\n"Root\\n(0.10 sec)";\n}\n', step_tree.to_pydot().to_string())

        # one child
        step_tree.add_step("FirstChild", "Root", self._step_with_runtime(1))
        self.assertEqual((
            'digraph G {\n'
            '"Root\\n(0.10 sec)";\n'
            '"Root\\n(0.10 sec)" -> "FirstChild\\n(1.00 sec)";\n'
            '}\n'
            ), step_tree.to_pydot().to_string())
