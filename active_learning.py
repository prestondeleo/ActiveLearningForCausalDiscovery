import data_generation as dg
import pc_algorithm as pc_a
import networkx as nx
import numpy as np
import pandas as pd
import random
from dowhy import CausalModel
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from causallearn.utils.GraphUtils import GraphUtils
from scipy import stats

class Experiment:
    def __init__(self, num_models:int, k:int)-> None:
        self.num_models = num_models
        self.k = k
        #self.sample_size = sample_size

    def get_num_nodes(self, cpdag:np.ndarray)->int:
        return len(cpdag)

    def get_neighbors(self, pcdag:np.ndarray, node:int)->np.ndarray:
        neighbors = np.where((pcdag[node, :] == 1))[0]
        return neighbors
    
    def intervene(self, data:pd.DataFrame, interv_node:int)->pd.DataFrame:
        intervened_data = data.copy()
        #intervened_data[interv_node] = 1
        intervened_data[interv_node] = 1
        #intervened_data[interv_node] = np.random.uniform(low=-10, high=10, size=len(data))
        """
        We need to change this so updated intervention data has children of node effected and not children not eeffected I beleieev
        """

        return intervened_data
    
    def discovery(self, pcdag:np.ndarray, interv_data:pd.DataFrame, interv_node:int, neighbor:int)->np.ndarray:
        pass


    def visualize_pcdag(adj_matrix: np.ndarray, pos=None, title="Generated Graph", figsize=(10, 8)):

        G = nx.DiGraph()
        n_nodes = adj_matrix.shape[0]
        G.add_nodes_from(range(n_nodes))
        
        for i in range(n_nodes):
            for j in range(n_nodes):
                if adj_matrix[i, j] == 1:
                    G.add_edge(i, j)
        
        if pos is None:
            pos = nx.spring_layout(G, seed=42)

        plt.figure(figsize=figsize)
        nx.draw(G, pos, with_labels=True, node_size=500, node_color='skyblue', 
                font_size=10, font_weight='bold', arrows=True)
        plt.title(title)
        plt.show()
        
        return pos  

    def random_dag_from_pcdag(pcdag_matrix):
        np.random.seed(seed = 47)  
        random.seed(47)
        G = nx.DiGraph(pcdag_matrix)
        
        undirected_edges = [(i, j) for i, j in G.edges if G.has_edge(j, i)]
        
        for u, v in undirected_edges:
            if G.has_edge(v, u):
                G.remove_edge(v, u)

        for u, v in undirected_edges:
            if not G.has_edge(u, v) and not G.has_edge(v, u):
                if random.choice([True, False]):  
                    G.add_edge(u, v)
                else:
                    G.add_edge(v, u)
            
            if not nx.is_directed_acyclic_graph(G):
                G.remove_edge(u, v)
                G.add_edge(v, u)
        return nx.to_numpy_array(G), G
    
if __name__ == '__main__':
        np.random.seed(seed = 47)  
        random.seed(47)
        G = dg.create_dag(n = 5, expected_degree = 1)
        start_adj_matrix = nx.to_numpy_array(G)        
        pcdag = pc_a.pc(G)
        experiment = Experiment(5, 5)
        true_DAG, DAG = experiment.random_dag_from_pcdag(pcdag) #gets random graph from MEC(s)
        data = dg.generate_data(graph = DAG, random_seed=47, num_rows=20000)
