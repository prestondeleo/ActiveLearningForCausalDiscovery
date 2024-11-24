import numpy as np
import pandas as pd
import random
import networkx as nx
import matplotlib.pyplot as plt
from CausalPlayground import CausalGraphGenerator, SCMGenerator, StructuralCausalModel


# creates a random DAG with weak connectivity. expected_degree ~ average number of edges per node
def create_dag(n: int, expected_degree: int) -> nx.DiGraph:
    # creates random spanning tree
    undirected_graph = nx.random_labeled_tree(n)

    dag = nx.DiGraph()
    dag.add_nodes_from(undirected_graph.nodes)

    # randomly assigns directions to spanning tree edges
    for u, v in undirected_graph.edges():
        if random.choice([True, False]):
            dag.add_edge(u, v)
            # reverses direction if adding edge creates a cycle
            if not nx.is_directed_acyclic_graph(dag):
                dag.remove_edge(u, v)
                dag.add_edge(v, u)
        else:
            dag.add_edge(v, u)
            if not nx.is_directed_acyclic_graph(dag):
                dag.remove_edge(v, u)
                dag.add_edge(u, v)

    max_edges = n * (n - 1) // 2  # max number of possible edges in a DAG
    num_edges = min(expected_degree * n / 2, max_edges)

    possible_edges = [(i, j) for i in range(n) for j in range(n) if i != j and not dag.has_edge(i, j)]
    random.shuffle(possible_edges)

    # randomly adds additional edges while maintaining acyclicity
    for source, target in possible_edges:
        if dag.number_of_edges() >= num_edges:
            break
        dag.add_edge(source, target)
        if not nx.is_directed_acyclic_graph(dag):
            dag.remove_edge(source, target)

    return dag


def display(graph: nx.DiGraph):
    plt.figure(figsize=(10, 8))
    nx.draw(graph, with_labels=True, node_size=500, node_color='skyblue', font_size=10, font_weight='bold',
            arrows=True)
    plt.title("Generated DAG")
    plt.show()


def save(graph: nx.DiGraph, expected_degree: int):
    plt.figure(figsize=(10, 8))
    nx.draw(graph, with_labels=True, node_size=500, node_color='skyblue', font_size=10, font_weight='bold',
            arrows=True)
    plt.title("Generated DAG")
    plt.savefig(f"dag_nodes_{graph.number_of_nodes()}_degree_{expected_degree}.png")
    plt.close()


# generates a dataframe of synthetic data based on a causal DAG
# coef_range = range for random coefficients, intercept_range = range for random intercepts
def generate_data(graph, num_rows=1000, coef_range=(0.5, 1.5), intercept_range=(-1.0, 1.0),
                  random_seed=None) -> pd.DataFrame:
    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("The input graph must be a directed acyclic graph (DAG).")

    if random_seed is not None:
        np.random.seed(random_seed)

    topo_order = list(nx.topological_sort(graph))

    # dictionary that stores generated data for each node
    data = {node: np.zeros(num_rows) for node in topo_order}

    for node in topo_order:
        parents = list(graph.predecessors(node))
        if not parents:
            data[node] = np.random.normal(0, 1, num_rows)
        else:
            coefficients = np.random.uniform(coef_range[0], coef_range[1], size=len(parents))
            intercept = np.random.uniform(intercept_range[0], intercept_range[1])

            # child node is a linear combination of parent values and intercept
            parent_values = np.array([data[parent] for parent in parents])
            data[node] = np.dot(coefficients, parent_values) + intercept

    # sorts the columns so that they are in numerical, not topological, order
    df = pd.DataFrame(data)
    return df[sorted(df.columns)]
