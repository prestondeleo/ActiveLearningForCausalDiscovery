from unittest import TestCase
import pytest
import networkx as nx

import data_generation


class Test(TestCase):
    def test_make_weakly_connected(self):
        """Test that a weakly connected graph remains unchanged."""
        G = nx.DiGraph()
        G.add_edges_from([(1, 2), (2, 3), (3, 4)])  # Already weakly connected
        original_edges = set(G.edges())

        data_generation.make_weakly_connected(G)
        self.assertEqual(set(G.edges()), original_edges)

        """Test that a disconnected DAG becomes weakly connected."""
        G = nx.DiGraph()
        G.add_edges_from([(1, 2), (3, 4)])  # Two disconnected components

        data_generation.make_weakly_connected(G)

        # Check that the graph is now weakly connected
        self.assertTrue(nx.is_weakly_connected(G))
        # Check that no cycles were introduced
        self.assertTrue(nx.is_directed_acyclic_graph(G))

        """Test a graph where components are already connected."""
        G = nx.DiGraph()
        G.add_edges_from([(1, 2), (2, 3), (3, 4)])  # Already connected
        original_edges = set(G.edges())

        data_generation.make_weakly_connected(G)
        self.assertEqual(set(G.edges()), original_edges)

        """Test a larger graph with multiple disconnected components."""
        G = nx.DiGraph()
        G.add_edges_from([(1, 2), (3, 4), (5, 6), (7, 8)])  # Four components

        data_generation.make_weakly_connected(G)

        # Check that the graph is now weakly connected
        self.assertTrue(nx.is_weakly_connected(G))
        # Check that no cycles were introduced
        self.assertTrue(nx.is_directed_acyclic_graph(G))

    def test_create_dag(self):
        with pytest.raises(ValueError):
            data_generation.create_dag(5, 0.01)  # Density too low to ensure weak connectivity

        def assert_properties(dag, expected_num_edges):
            assert nx.is_directed_acyclic_graph(dag), "Graph is not acyclic."
            assert nx.is_weakly_connected(dag), "Graph is not weakly connected."
            assert dag.number_of_edges() == expected_num_edges, "Graph has incorrect edge count."

        # low density graph
        assert_properties(data_generation.create_dag(50, 0.1), 245)

        # high density graph
        assert_properties(data_generation.create_dag(50, 0.5), 1225)