# import guacml as guac
# import os
# import pandas as pd
import unittest

from guacml.step_tree.step_tree import StepTree
from guacml.step_tree.base_step import BaseStep


class TestStepTree(unittest.TestCase):
    def test_to_dot(self):
        # empty tree gives empty result
        step_tree = StepTree("target", 0, "eval_metric")
        self.assertEqual("digraph G {\n\t\n}\n", step_tree.to_dot())

        # one node tree
        step_tree.add_step("Root", None, BaseStep())
        self.assertEqual("digraph G {\n\tRoot\n}\n", step_tree.to_dot())

        # one child
        step_tree.add_step("FirstChild", "Root", BaseStep())
        self.assertEqual("digraph G {\n\tRoot\n\tRoot -> FirstChild\n}\n", step_tree.to_dot())

        # child with childs
        step_tree.add_step("SecondChild", "Root", BaseStep())
        step_tree.add_step("FirstGrandChild", "FirstChild", BaseStep())
        step_tree.add_step("SecondGrandChild", "FirstChild", BaseStep())
        self.assertEqual((
            "digraph G {\n"
            "\tRoot\n"
            "\tRoot -> FirstChild\n"
            "\tRoot -> SecondChild\n"
            "\tFirstChild -> FirstGrandChild\n"
            "\tFirstChild -> SecondGrandChild\n"
            "}\n"
            ), step_tree.to_dot())

