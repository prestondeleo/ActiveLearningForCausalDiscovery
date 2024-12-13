import numpy as np
import pandas as pd
import random
import networkx as nx
import matplotlib.pyplot as plt
#from CausalPlayground import CausalGraphGenerator, SCMGenerator, StructuralCausalModel


# creates a random DAG with weak connectivity. expected_degree ~ average number of edges per node
def create_dag(n: int, expected_degree: int) -> nx.DiGraph:
    # creates random spanning tree
    undirected_graph = nx.random_labeled_tree(n)

    DAG = nx.DiGraph()
    DAG.add_nodes_from(undirected_graph.nodes)

    # creates immorality w/ 3 randomly selected nodes
    nodes = list(undirected_graph.nodes)
    random.shuffle(nodes)
    a, b, c = nodes[:3]
    DAG.add_edge(a, b)
    DAG.add_edge(c, b)

    # excludes these edges to preserve immorality and acyclicity
    excluded_edges = {(a, c), (c, a), (b, a), (b, c)}

    max_edges = n * (n - 1) // 2  # max number of possible edges in a DAG
    num_edges = min(expected_degree * n / 2, max_edges)

    # randomly assigns directions to spanning tree edges
    for u, v in undirected_graph.edges():
        if DAG.number_of_edges() >= num_edges:
            break
        if random.choice([True, False]) and (u,v) not in excluded_edges:
            DAG.add_edge(u, v)
            # reverses direction if adding edge creates a cycle
            if not nx.is_directed_acyclic_graph(DAG):
                DAG.remove_edge(u, v)
                DAG.add_edge(v, u)
        elif (v,u) not in excluded_edges:
            DAG.add_edge(v, u)
            if not nx.is_directed_acyclic_graph(DAG):
                DAG.remove_edge(v, u)
                DAG.add_edge(u, v)

    possible_edges = [(i, j) for i in range(n) for j in range(n) if i != j and not DAG.has_edge(i, j)]
    possible_edges = [edge for edge in possible_edges if edge not in excluded_edges]
    random.shuffle(possible_edges)

    # randomly adds additional edges while maintaining acyclicity
    for source, target in possible_edges:
        if DAG.number_of_edges() >= num_edges:
            break
        DAG.add_edge(source, target)
        if not nx.is_directed_acyclic_graph(DAG):
            DAG.remove_edge(source, target)

    return DAG


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
    assert nx.is_directed_acyclic_graph(graph)

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

# adj_matrix is for a CPDAG
def remove_undirected_edges_from(adj_matrix: np.ndarray):
    undirected_edges = (adj_matrix == 1) & (adj_matrix.T == 1)

    # Remove symmetric edges by setting them to 0
    adj_matrix[undirected_edges] = 0

    return adj_matrix

# selects a random connected subset of k nodes from the graph represented by adjacency matrix A
# k = desired number of nodes
def select_random_subgraph_from(A, k):
    num_nodes = A.shape[0]

    # randomly selects starting node
    start_node = np.random.choice(num_nodes)
    selected_nodes = [start_node]

    # candidates for next node to add
    possible_candidates = set(np.where(A[start_node] > 0)[0])

    while len(selected_nodes) < k:
        if not possible_candidates:
            raise ValueError("Cannot expand subset further while maintaining connectivity.")

        new_node = np.random.choice(list(possible_candidates))

        selected_nodes.append(new_node)
        possible_candidates.remove(new_node) # doesn't allow a node to be added twice

        new_possible_neighbors = set(np.where(A[new_node] > 0)[0])
        possible_candidates.update(new_possible_neighbors - set(selected_nodes))

    return A[np.ix_(selected_nodes, selected_nodes)], selected_nodes

