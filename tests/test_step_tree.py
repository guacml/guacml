# import guacml as guac
# import os
# import pandas as pd
import unittest

from guacml.step_tree.step_tree import StepTree
from guacml.step_tree.base_step import BaseStep


class TestStepTree(unittest.TestCase):
    def test_to_pydot(self):
        # empty tree gives empty result
        step_tree = StepTree("target", 0, "eval_metric")
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
