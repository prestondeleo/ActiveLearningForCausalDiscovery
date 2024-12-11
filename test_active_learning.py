import networkx as nx
from networkx.utils.misc import graphs_equal

import data_generation
import pc_algorithm
from active_learning import Experiment

def test_total_full_discovery():
    G = data_generation.create_dag(n=10, expected_degree=3)
    pcdag = pc_algorithm.pc(G)
    experiment = Experiment(5, 5)
    true_DAG, DAG = experiment.random_dag_from_pcdag(pcdag)  # gets random graph from MEC(s)
    data = data_generation.generate_data(graph=DAG, random_seed=47, num_rows=20000)

    predicted_adj_matrix = experiment.total_full_discovery(true_causal_graph=DAG, pcdag=pcdag, data=data)
    predicted_dag = nx.from_numpy_array(predicted_adj_matrix)

    experiment.visualize_pcdag(predicted_adj_matrix)

    assert nx.is_directed_acyclic_graph(predicted_dag), "Graph is not acyclic."
    assert nx.is_weakly_connected(predicted_dag), "Graph is not weakly connected."
    assert nx.number_of_selfloops(predicted_dag) == 0, "There are self loops."

    assert (set(G.nodes) == set(predicted_dag.nodes)) and (set(G.edges) == set(predicted_dag.edges))