import networkx as nx
import numpy as np

import data_generation
import pc_algorithm
from active_learning import Experiment

def test_random_dag_from_pcdag():
    G = data_generation.create_dag(n=10, expected_degree=3)
    pcdag = pc_algorithm.pc(G)
    experiment = Experiment(5, 5)
    true_DAG, DAG = experiment.random_dag_from_pcdag(pcdag)  # gets random graph from MEC(s)

    assert nx.is_directed_acyclic_graph(DAG), "Graph is not acyclic."
    assert nx.is_weakly_connected(DAG), "Graph is not weakly connected."
    assert nx.number_of_selfloops(DAG) == 0, "There are self loops."

def test_total_full_discovery():
    G = data_generation.create_dag(n=10, expected_degree=3)
    pcdag = pc_algorithm.pc(G)
    experiment = Experiment(5, 5)
    true_DAG, DAG = experiment.random_dag_from_pcdag(pcdag)  # gets random graph from MEC(s)
    data = data_generation.generate_data(graph=DAG, random_seed=47, num_rows=20000)

    predicted_adj_matrix = experiment.total_full_discovery(true_causal_graph=DAG, pcdag=pcdag, data=data)
    predicted_dag = nx.from_numpy_array(predicted_adj_matrix, create_using=nx.DiGraph)

    # tests that the 2 graphs are equal
    assert np.array_equal(true_DAG, predicted_adj_matrix)

    assert nx.is_directed_acyclic_graph(predicted_dag), "Graph is not acyclic."
    assert nx.is_weakly_connected(predicted_dag), "Graph is not weakly connected."
    assert nx.number_of_selfloops(predicted_dag) == 0, "There are self loops."

def test_rand_subsam_w_rep():
    G = data_generation.create_dag(10, 3)
    pcdag = pc_algorithm.pc(G)

    experiment = Experiment(5, 5)
    true_DAG, DAG = experiment.random_dag_from_pcdag(pcdag)

    subsample_adj_matrices = experiment.rand_subsam_w_rep(true_DAG)

    for adj_matrix in subsample_adj_matrices:
        subsample_graph = nx.from_numpy_array(adj_matrix, create_using = nx.DiGraph)

        #experiment.visualize_pcdag(adj_matrix)

        assert nx.is_directed_acyclic_graph(subsample_graph), "Graph is not acyclic."
        assert nx.number_of_selfloops(subsample_graph) == 0, "There are self loops."

        # checks that A is a subgraph of B
        nodes_match = set(subsample_graph.nodes).issubset(set(DAG.nodes))
        edges_match = set(subsample_graph.edges).issubset(set(DAG.edges))
        assert nodes_match and edges_match