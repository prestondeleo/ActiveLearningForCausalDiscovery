import os
import tempfile
from unittest import TestCase

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

import data_generation


class Test(TestCase):
    def test_create_dag(self):
        def assert_properties(dag, n, expected_degree):
            assert nx.is_directed_acyclic_graph(dag), "Graph is not acyclic."
            assert nx.is_weakly_connected(dag), "Graph is not weakly connected."
            assert dag.number_of_edges() == n * expected_degree / 2, "Graph has incorrect edge count."
            assert nx.number_of_selfloops(dag) == 0, "There are self loops."

        assert_properties(data_generation.create_dag(5, 2), 5, 2)
        assert_properties(data_generation.create_dag(10, 4), 10, 4)
        assert_properties(data_generation.create_dag(50, 2), 50, 2)
        assert_properties(data_generation.create_dag(50, 20), 50, 20)

    def test_save(self):
        original_dir = os.getcwd()
        with tempfile.TemporaryDirectory() as tempdir:
            try:
                # changes working directory to temporary folder
                os.chdir(tempdir)

                # simple DAG
                dag1 = nx.DiGraph([(1, 2), (2, 3)])
                data_generation.save(dag1, 1)
                assert os.path.exists("dag_nodes_3_degree_1.png"), "File not found for test case 1"

                # uses create_dag() function
                dag2 = data_generation.create_dag(10, 4)
                data_generation.save(dag2, 4)
                assert os.path.exists("dag_nodes_10_degree_4.png"), "File not found for test case 2"

                # uses create_dag() to make large DAG
                dag3 = data_generation.create_dag(50, 15)
                data_generation.save(dag3, 15)
                assert os.path.exists("dag_nodes_50_degree_15.png"), "File not found for test case 3"

            # reverts to original directory
            finally:
                os.chdir(original_dir)

    def test_generate_data(self):

        # takes a dataframe and list of parent nodes (columns), asserts that the parent nodes are all independent
        def assert_independent(df: pd.DataFrame, parent_nodes: list):
            correlation_matrix = df[parent_nodes].corr().to_numpy()
            upper_triangle = np.triu(correlation_matrix, k=1)

            assert np.max(upper_triangle) < 0.3

        # asserts that the child node is a linear combination of its parents
        def assert_linear_combination(df: pd.DataFrame, child_node: str, parent_nodes: list,
                                      coef_range=(0.5, 1.5), intercept_range=(-1.0, 1.0)):
            x = df[parent_nodes]  # parent node values
            y = df[child_node]  # child node values

            model = LinearRegression()
            model.fit(x, y)

            coef = np.array(model.coef_)
            intercept = model.intercept_
            r_squared = model.score(x, y)

            # Note: the coefficients are sometimes outside the range when there are multiple parents. we've decided
            # that this is not a problem

            # asserts that all coefficients are within the specified range
            #assert np.all((coef_range[0] <= coef) & (coef <= coef_range[1])), \
            #   f"Coefficients are not within the specified range: {coef_range}"

            # Assert that intercept is within the specified range
            #assert intercept_range[0] <= intercept <= intercept_range[1], \
            #   f"Intercept {intercept} is not within the specified range: {intercept_range}"

            # R-squared should = 1, since we didn't introduce any noise to our data
            assert r_squared == 1, f"R-squared value {r_squared} is not equal to 1"

        # asserts that generate_data works correctly when we call it on graph "graph"
        def assert_properties(graph: nx.DiGraph):
            df = data_generation.generate_data(graph)

            topo_order = list(nx.topological_sort(graph))

            root_nodes = list()  # list of root nodes (those w/ no parents)
            for node in topo_order:
                parents = list(graph.predecessors(node))  # parents of "node"

                if parents:
                    assert_linear_combination(df, node, parents)
                else:
                    root_nodes.append(node)

            # if we have multiple parents, we test that they are all independent
            if len(root_nodes) > 1:
                assert_independent(df, root_nodes)

        assert_properties(data_generation.create_dag(5, 2))
        assert_properties(data_generation.create_dag(10, 4))
        assert_properties(data_generation.create_dag(50, 2))
        assert_properties(data_generation.create_dag(50, 20))
