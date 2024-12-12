from typing import List, Any

from numpy import ndarray, dtype

import data_generation as dg
import pc_algorithm as pc_a
import networkx as nx
import numpy as np
import pandas as pd
import random
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

    # gives interventional data for single vertex intervention
    def intervene(self, true_causal_graph: nx.DiGraph, data:pd.DataFrame, interv_node:int)->pd.DataFrame:
        intervened_data = data.copy()

        intervened_data[interv_node] = 1

        # children of intervened node
        child_indices = [int(child) for child in true_causal_graph.successors((interv_node))]

        # multiplies values of children of intervened node by 100
        #child_columns = intervened_data.iloc[:, child_indices]
        intervened_data.iloc[:, child_indices] *= 100

        return intervened_data
    
    def discovery(self, pcdag:np.ndarray, interv_data:pd.DataFrame, neighbor_interv_data:pd.DataFrame, interv_node:int, neighbor:int)->np.ndarray:
        model_1 = LinearRegression()
        model_1.fit(interv_data[[interv_node]], interv_data[neighbor])
        res_1 = interv_data[neighbor] - model_1.predict(interv_data[[interv_node]])

        model_2 = LinearRegression()
        
        model_2.fit(neighbor_interv_data[[neighbor]], neighbor_interv_data[interv_node])
        res_2 = neighbor_interv_data[interv_node] - model_2.predict(interv_data[[neighbor]])
        
  
        if np.var(res_1) > np.var(res_2):
            pcdag[interv_node, neighbor] = 1
            pcdag[neighbor, interv_node] = 0

        else:
            pcdag[interv_node, neighbor] = 0
            pcdag[neighbor, interv_node] = 1
        return pcdag
    
    def total_full_discovery(self, true_causal_graph: nx.DiGraph, pcdag:np.ndarray, data:pd.DataFrame)->np.ndarray:
        new_pcdag = pcdag.copy()
        for interv_node in range(pcdag.shape[0]):
                interv_data = self.intervene(true_causal_graph, data, interv_node)
                #neighbor_interv_data = self.intervene(true_causal_graph, interv_node, data)
                #print('interv node')
                #print(interv_node)
                neighbors = self.get_neighbors(pcdag, interv_node)
                #print('neighbors')
                #print(neighbors)
                for neighbor in neighbors:
                    neighbor_interv_data = self.intervene(true_causal_graph, data, neighbor)

                    if pcdag[interv_node, neighbor] == 1 and  pcdag[neighbor, interv_node] == 1:

                        new_pcdag = self.discovery(pcdag=new_pcdag.copy(), interv_data = interv_data, neighbor_interv_data = neighbor_interv_data, interv_node=interv_node, neighbor=neighbor)
        return new_pcdag

    def visualize_pcdag(self, adj_matrix: np.ndarray, pos=None, title="Generated Graph", figsize=(10, 8)):

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

    def random_dag_from_pcdag(self, pcdag_matrix):
        np.random.seed(seed = 47)  
        random.seed(47)
        G = nx.DiGraph(pcdag_matrix)

        # adds i<j so that we only include each undirected edge once
        undirected_edges = [(i, j) for i, j in G.edges if G.has_edge(j, i) and i<j]
        
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
            #We start with the cpdag
        #we remove all teh undirected edges
        # we are left with a DAG now
        #We make subgraphs from this new DAG
        # we unorient edges in subgraphs and make this training data i.e. sunbgraph with some undirected edges and the DAG subgraph

        #asfter this training we try to infer the original PCDAG and make a DAG


#def rand_subsam_w_rep(self, cpdag:np.ndarray,  num_nodes:int, min_perc_samp = 0.25, max_perc_samp = 0.9)    
    def rand_subsam_w_rep(self, cpdag:np.ndarray)-> list[ndarray[Any, dtype[Any]]]:



        num_nodes = self.get_num_nodes(cpdag)
        #np.random.seed(seed = 47)  
        #random.seed(47)
        G = nx.DiGraph(cpdag)
        undirected_edges = [(i, j) for i, j in G.edges if G.has_edge(j, i) and i<j]
        for u, v in undirected_edges:
            if G.has_edge(v, u):
                G.remove_edge(v, u)

        original_num_isolated_nodes = nx.number_of_isolates(G)
        edges = list(G.edges)
        #print('edges')
        #print(len(edges))
        #print(original_num_isolated_nodes)
        subgraphs = []
        subset_size = np.random.randint(2,  len(edges) + 1)
        successful_draws = 0

        while successful_draws != subset_size:
            subset_size_graph = np.random.randint(2,  len(edges) + 1)
            #sampled_edges = random.choices(edges, k=subset_size_graph)#This might be doing with replacement
            sampled_edge_indices = np.random.choice(len(edges), size=subset_size_graph, replace=False)
            sampled_edges = [edges[i] for i in sampled_edge_indices]

            subgraph = nx.DiGraph()
            subgraph.add_nodes_from(G.nodes())
            subgraph.add_edges_from(sampled_edges)
            subgraph_num_isolated_nodes = nx.number_of_isolates(subgraph)
            
            if subgraph_num_isolated_nodes != original_num_isolated_nodes: #maybe its okay to have less nodes in subgraph
                print(subgraph_num_isolated_nodes)
                print(original_num_isolated_nodes)
                continue
                
            else:
                subgraph_adj_matrix = nx.to_numpy_array(subgraph, nodelist=range(num_nodes))  # Maintain node index consistency
                subgraphs.append(subgraph_adj_matrix)
                successful_draws += 1
                print(successful_draws)

        return subgraphs
    
    def model_train_data(self, cpdag:np.ndarray)->tuple:
        pass



        #sample size would be number of nodes. 
        #could be called subset size
        """

        num_nodes = self.get_num_nodes(cpdag)
        #subset_indices
        subset_size = np.random.randint(int(min_perc_samp * num_nodes),  int(max_perc_samp * num_nodes) + 1  )

        sampled_indices = np.random.choice(num_nodes, size=subset_size, replace=True)

        #subgraph_adjmatrix = np.ix_()
        #Does not gurantee all nodes will be connected to at least one other node (single vertex with no edge)
        subgraph_adjmatrix = cpdag.ix_(sampled_indices,sampled_indices)
        """
        #return subgraph_adjmatrix

"""
if __name__ == '__main__':
        np.random.seed(seed = 47)  
        random.seed(47)
        G = dg.create_dag(n = 20, expected_degree = 1)
        start_adj_matrix = nx.to_numpy_array(G)        
        pcdag = pc_a.pc(G)
        experiment = Experiment(5, 5)
        shared_pos = experiment.visualize_pcdag(pcdag, title="PCDAG")
        true_DAG, DAG = experiment.random_dag_from_pcdag(pcdag) #gets random graph from MEC(s)

        experiment.visualize_pcdag(true_DAG, pos=shared_pos, title="true DAG")

        
        #P1
        true_DAG, DAG = experiment.random_dag_from_pcdag(pcdag) #gets random graph from MEC(s)
        data = dg.generate_data(graph = DAG, random_seed=47, num_rows=20000)

        shared_pos = experiment.visualize_pcdag(pcdag, title="PCDAG")

        experiment.visualize_pcdag(true_DAG, pos=shared_pos, title="true DAG")

        #print([int(child) for child in DAG.successors(8)])

        #intervened_data = experiment.intervene(DAG, data, 8)
        #print(intervened_data.to_string())

        predicted_adj_matrix = experiment.total_full_discovery(true_causal_graph = DAG, pcdag=pcdag, data=data)

        experiment.visualize_pcdag(predicted_adj_matrix, pos=shared_pos, title="predicted DAG")

        test_mat, test_dag = experiment.rand_subsam_w_rep(cpdag = pcdag, num_nodes=4)
        experiment.visualize_pcdag(adj_matrix=test_mat, pos=shared_pos)
        #""end p1
        
        sample_subgraphs = experiment.rand_subsam_w_rep(cpdag = pcdag)

        for subgraph in sample_subgraphs:
            print(len(subgraph))
            experiment.visualize_pcdag(subgraph, pos=shared_pos, title="true DAG")
        """

    