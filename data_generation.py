import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt
from CausalPlayground import CausalGraphGenerator, SCMGenerator, StructuralCausalModel
from typing import Callable, Dict

def create_dag(n: int, density: float) -> nx.DiGraph:
    """
    Creates a Directed Acyclic Graph (DAG) with the specified number of variables, and a given density.

    Parameters:
    - n (int): Number of endogenous variables (internal nodes).
    - density (float): Desired density

    Returns:
    - nx.DiGraph: A directed acyclic graph meeting the specified constraints.
    """

    # density can't > 0.5 for a DAG
    assert 0<density<=0.5

    # Calculate the maximum number of edges possible in a graph with n nodes
    max_edges = n * (n - 1)

    # Calculate the number of edges needed for the specified density
    num_edges = int(density * max_edges)

    if n-1 > num_edges:
        raise ValueError("The desired density is too small to achieve weak connectivity.")


    nodes = [f"X{i}" for i in range(n)]


    # Create an empty directed graph
    dag = nx.DiGraph()
    dag.add_nodes_from(nodes)

    used_edges=set()

    # Ensure weak connectivity and acyclicity by connecting each node to a random node later in the topological order
    for i in range(n - 1):
        target = random.choice(nodes[i + 1:])
        dag.add_edge(nodes[i], target)
        used_edges.add((nodes[i], target))


    # Generate all possible edges while preserving acyclicity and ensuring that we don't duplicate any edges
    possible_edges = [
        (source, target) for i, source in enumerate(nodes)
        for target in nodes[i + 1:]
        if (source, target) not in used_edges
    ]
    random.shuffle(possible_edges)

    # Add edges to reach the desired density
    for source, target in possible_edges:
        if dag.number_of_edges() >= num_edges:
            break
        dag.add_edge(source, target)

    return dag


def display(graph: nx.DiGraph):
    plt.figure(figsize=(10, 8))
    nx.draw(graph, with_labels=True, node_size=500, node_color='skyblue', font_size=10, font_weight='bold',
            arrows=True)
    plt.title("Generated DAG")
    plt.show()

def create_scms(n_endo: int, p: float):
    graph=create_dag(n_endo, p)

    # Define a dictionary of functions
    all_functions: Dict[str, Callable] = {
        "random_linear": lambda x: sum(random.uniform(-10, 10) * xi for xi in x) + random.uniform(-10, 10)
    }

    scm_generator=SCMGenerator(all_functions=all_functions, seed=6)
    scm=scm_generator.create_scm_from_graph(graph=graph, possible_functions=["random_linear"],
                                            exo_distribution=np.random.normal,
                                            exo_distribution_kwargs={"loc": 0, "scale": 1}) # mean 0, std deviation 1
    print(scm)


dag=create_dag(20, 0.2)