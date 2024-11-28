import networkx as nx
import numpy as np

# PC algorithm step 1: returns the skeleton of a DAG in the form of an adjacency matrix
def skeleton(dag: nx.DiGraph) -> np.ndarray:
    assert nx.is_directed_acyclic_graph(dag)

    # Create an adjacency matrix of size (n_nodes x n_nodes)
    n = dag.number_of_nodes()
    adj_matrix = np.zeros((n, n), dtype=int)

    # Get node list sorted for consistent indexing
    nodes = sorted(dag.nodes)
    node_index = {node: idx for idx, node in enumerate(nodes)}

    # Extract the edges from the DAG
    edges = dag.edges()

    # Get the indices of the nodes in the edges
    indices_u = np.array([node_index[u] for u, _ in edges], dtype=int)
    indices_v = np.array([node_index[v] for _, v in edges], dtype=int)

    # Set the undirected edges in the adjacency matrix
    adj_matrix[indices_u, indices_v] = 1
    adj_matrix[indices_v, indices_u] = 1

    return adj_matrix

# PC algorithm step 2: updates the skeleton adjacency matrix to reflect immoralities
def add_immoralities(dag: nx.DiGraph, skeleton: np.ndarray) -> np.ndarray:
    assert nx.is_directed_acyclic_graph(dag)

    # returns set of tuples (X, Z, Y) in dag where there is no edge between X and Y, and Z was not in the conditioning
    # set that makes X and Y conditionally independent
    immoralities=list(nx.algorithms.dag.v_structures(dag))

    # if there are no immoralities, we don't have to update the adjacency matrix
    if not immoralities:
        return skeleton

    # Create a node index mapping once
    nodes=sorted(dag.nodes)
    node_index = {node: idx for idx, node in enumerate(nodes)}

    # Convert immoralities into arrays for efficient indexing
    X, Z, Y = zip(*immoralities)  # Unpack tuples (X, Z, Y)

    x_indexes = np.array([node_index[x] for x in X], dtype=int)
    z_indexes = np.array([node_index[z] for z in Z], dtype=int)
    y_indexes = np.array([node_index[y] for y in Y], dtype=int)

    # we have the undirected edges X-Z and Y-Z from the skeleton graph, so we just need to remove (Z,X) and (Z,Y) to
    # make the edges directed
    skeleton[z_indexes, x_indexes] = 0
    skeleton[z_indexes, y_indexes] = 0

    return skeleton

# PC algorithm step 3: returns the final adjacency matrix for the essential graph
def orient_edges_incident_on_colliders(adjacency_matrix: np.ndarray) -> np.ndarray:
    n = adjacency_matrix.shape[0]

    # Iterate over all possible triplets (X, Z, Y)
    for X in range(n):
        for Z in range(n):
            if adjacency_matrix[X][Z] == 1 and adjacency_matrix[Z][X] == 0:  # Directed edge X → Z
                for Y in range(n):
                    # Undirected edge Z — Y
                    if X != Y and Z != Y and adjacency_matrix[Z][Y] == 1 and adjacency_matrix[Y][Z] == 1:
                        # Ensure no edge between X and Y
                        if adjacency_matrix[X][Y] == 0 and adjacency_matrix[Y][X] == 0:
                            adjacency_matrix[Y][Z]=0 # creates a directed edge (Z,Y)

    return adjacency_matrix

# returns an adjacency matrix representing the essential graph. matrix[a][b] means that node a is adjacent to node b.
# if (a,b) is a directed edge, matrix[a][b]=1 and matrix[b][a]=0. if (a,b) is undirected, matrix[a][b]=matrix[b][a]=1
def pc(dag: nx.DiGraph) -> np.ndarray:
    skeleton_adj_matrix=skeleton(dag)
    adj_matrix_with_immoralities=add_immoralities(dag, skeleton_adj_matrix)
    return orient_edges_incident_on_colliders(adj_matrix_with_immoralities)