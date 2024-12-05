import numpy as np

def is_acyclic(adj_matrix: np.ndarray) -> bool:
    # number of nodes in the graph
    n = len(adj_matrix)

    # by definition, we need at least 3 nodes to have a cycle
    if n<3:
        return True

    def dfs(node, visited, in_recursion_stack):
        visited[node] = True
        in_recursion_stack[node] = True # nodes being explored in the current DFS path

        for neighbor in range(n):
            if adj_matrix[node][neighbor] == 1 and node!=neighbor:
                # there is a cycle if the neighbor is in the recursion stack
                if in_recursion_stack[neighbor]:
                    return False

                # explores neighbor if it is not visited
                if not visited[neighbor]:
                    if not dfs(neighbor, visited, in_recursion_stack):
                        return False

        # removes node from recursion stack
        in_recursion_stack[node] = False
        return True

    visited = [False] * n
    in_recursion_stack = [False] * n

    # runs dfs from the first node, since we know the graph is connected
    return dfs(0, visited, in_recursion_stack)

def is_acyclic2(adj_matrix: np.ndarray) -> bool:
    # number of nodes in the graph
    n = len(adj_matrix)

    # by definition, we need at least 3 nodes to have a cycle
    if n < 3:
        return True

    def dfs(node, visited, in_recursion_stack):
        #if not visited[node]:
        visited[node] = True
        in_recursion_stack[node] = True  # nodes being explored in the current DFS path

        for neighbor in range(n):
            if adj_matrix[node][neighbor] == 1 and node != neighbor:
                # there is a cycle if the neighbor is in the recursion stack
                if in_recursion_stack[neighbor]:
                    return False

                # explores neighbor if it is not visited
                if not visited[neighbor]:
                    if not dfs(neighbor, visited, in_recursion_stack):
                        return False

        # removes node from recursion stack
        in_recursion_stack[node] = False
        return True

    visited = [False] * n
    in_recursion_stack = [False] * n

    # runs dfs from the first node, since we know the graph is connected
    return dfs(0, visited, in_recursion_stack)