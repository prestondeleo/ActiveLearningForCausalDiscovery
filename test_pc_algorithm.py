import pc_algorithm
import networkx as nx
import numpy as np
from unittest import TestCase


class Test(TestCase):
    def test_skeleton(self):
        # Empty DAG (no nodes, no edges)
        dag = nx.DiGraph()
        expected = np.array([]).reshape(0, 0)  # Empty adjacency matrix
        result = pc_algorithm.skeleton(dag)
        assert np.array_equal(result, expected)

        # DAG with one node and no edges
        dag2 = nx.DiGraph()
        dag2.add_node(0)
        expected2 = np.array([[0]])  # Single node, no edges
        result2 = pc_algorithm.skeleton(dag2)
        assert np.array_equal(result2, expected2)

        # DAG with multiple edges (0 -> 1, 1 -> 2, 2 -> 3, 1 -> 3)
        dag3 = nx.DiGraph()
        dag3.add_edges_from([(0, 1), (1, 2), (2, 3), (1, 3)])
        expected3 = np.array([[0, 1, 0, 0], [1, 0, 1, 1], [0, 1, 0, 1], [0, 1, 1, 0]])
        result3 = pc_algorithm.skeleton(dag3)
        assert np.array_equal(result3, expected3)

    def test_add_immoralities(self):
        def adj_matrix_with_immoralities(dag):
            adj_matrix = pc_algorithm.skeleton(dag)
            return pc_algorithm.add_immoralities(dag, adj_matrix)

        # graph w/ no immoralities
        dag_1 = nx.DiGraph()
        dag_1.add_edges_from([(0, 1), (1, 2), (2, 3)])
        adj_matrix_1 = adj_matrix_with_immoralities(dag_1)

        # No immoralities, the adjacency matrix should be the same as that of the skeleton graph
        expected_matrix_1 = np.array([[0, 1, 0, 0],
                                      [1, 0, 1, 0],
                                      [0, 1, 0, 1],
                                      [0, 0, 1, 0]], dtype=int)
        assert np.array_equal(adj_matrix_1, expected_matrix_1)

        # graph w/ 1 immorality
        dag_2 = nx.DiGraph()
        dag_2.add_edges_from([(1, 2), (0, 2)])  # immorality at (1, 2, 0)
        adj_matrix_2 = adj_matrix_with_immoralities(dag_2)

        # matrix should no longer be symmetrical since we are making edges directed
        expected_matrix_2 = np.array([[0, 0, 1],
                                      [0, 0, 1],
                                      [0, 0, 0]], dtype=int)
        assert np.array_equal(adj_matrix_2, expected_matrix_2)

        # graph w/ multiple immoralities
        dag_3 = nx.DiGraph()
        dag_3.add_edges_from([(1,0), (4,0), (1,2), (3,2)])
        adj_matrix_3 = adj_matrix_with_immoralities(dag_3)

        # matrix should no longer be symmetrical since we are making edges directed
        expected_matrix_3 = np.array([[0, 0, 0, 0, 0],
                                      [1, 0, 1, 0, 0],
                                      [0, 0, 0, 0, 0],
                                      [0, 0, 1, 0, 0],
                                      [1, 0, 0, 0, 0]], dtype=int)
        assert np.array_equal(adj_matrix_3, expected_matrix_3)

    def test_orient_edges_incident_on_colliders(self):
        # DAG w/ 1 immorality
        dag=nx.DiGraph()
        dag.add_edges_from([(0, 2), (1, 2), (2,3), (2,4)])

        skeleton=pc_algorithm.skeleton(dag)
        adj_matrix_with_immoralities=pc_algorithm.add_immoralities(dag, skeleton)
        essential_graph_adj_matrix=pc_algorithm.orient_edges_incident_on_colliders(adj_matrix_with_immoralities)

        expected_matrix=np.array([[0, 0, 1, 0, 0],
                                  [0, 0, 1, 0, 0],
                                  [0, 0, 0, 1, 1],
                                  [0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0]])

        assert np.array_equal(essential_graph_adj_matrix, expected_matrix)

    def test_pc(self):
        dag = nx.DiGraph()
        dag.add_edges_from([(0, 2), (1, 2), (2, 3), (2, 4)])

        expected_matrix = np.array([[0, 0, 1, 0, 0],
                                    [0, 0, 1, 0, 0],
                                    [0, 0, 0, 1, 1],
                                    [0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0]])

        assert np.array_equal(pc_algorithm.pc(dag), expected_matrix)