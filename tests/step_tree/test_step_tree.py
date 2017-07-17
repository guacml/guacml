import unittest

from guacml.step_tree.step_tree import StepTree
from guacml.step_tree.base_step import BaseStep


class TestStepTree(unittest.TestCase):
    def setUp(self):
        self.config = {'run_time': {'inplace'}}

    def _new_empty_tree(self):
        empty_conf = {}
        return StepTree(empty_conf, None)

    def _step_with_runtime(self, runtime):
        step = BaseStep(None)
        step.runtime = runtime
        return step

    # ToDo: This test fails non deterministically
    def xtest_to_pydot(self):
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
        self.assertEqual(
            (
                'digraph G {\n'
                '"Root\\n(0.10 sec)";\n'
                '"Root\\n(0.10 sec)" -> "FirstChild\\n(1.00 sec)";\n'
                '}\n'
            ),
            step_tree.to_pydot().to_string()
        )

    def test_add_step(self):
        tree = StepTree(None, None)
        tree.add_step('step_1', None, BaseStep(self.config))
        tree.add_step('step_2', 'step_1', BaseStep(self.config))
        tree.add_step('step_3', 'step_1', BaseStep(self.config))

        self.assertListEqual(tree.get_children('step_1'), ['step_2', 'step_3'])
        self.assertListEqual(tree.get_leaf_names(), ['step_2', 'step_3'])

    def test_insert_step_before_root(self):
        tree = StepTree(None, None)
        tree.add_step('step_0', None, BaseStep(self.config))
        tree.add_step('step_1', 'step_0', BaseStep(self.config))

        inserted_step = BaseStep(self.config)
        tree.insert_step_before('inserted', 'step_0', inserted_step)

        self.assertEqual(tree.get_step('inserted'), inserted_step)
        self.assertListEqual(tree.get_children('inserted'), ['step_0'])
        self.assertListEqual(tree.get_children('step_0'), ['step_1'])
        self.assertListEqual(tree.get_leaf_names(), ['step_1'])

    def test_insert_step_before(self):
        tree = StepTree(None, None)
        tree.add_step('step_0', None, BaseStep(self.config))
        tree.add_step('step_1', 'step_0', BaseStep(self.config))

        inserted_step = BaseStep(self.config)
        tree.insert_step_before('inserted', 'step_1', inserted_step)

        self.assertEqual(tree.get_step('inserted'), inserted_step)
        self.assertListEqual(tree.get_children('step_0'), ['inserted'])
        self.assertListEqual(tree.get_children('inserted'), ['step_1'])
        self.assertListEqual(tree.get_leaf_names(), ['step_1'])

    def test_insert_step_after(self):
        tree = StepTree(None, None)
        tree.add_step('step_0', None, BaseStep(self.config))
        tree.add_step('step_1', 'step_0', BaseStep(self.config))
        tree.add_step('step_2', 'step_0', BaseStep(self.config))

        inserted_step = BaseStep(self.config)
        tree.insert_step_after('inserted', 'step_0', inserted_step)

        self.assertEqual(tree.get_step('inserted'), inserted_step)
        self.assertListEqual(tree.get_children('step_0'), ['inserted'])
        self.assertListEqual(tree.get_children('inserted'), ['step_1', 'step_2'])
        self.assertListEqual(tree.get_leaf_names(), ['step_1', 'step_2'])
