from typing import List, Any

from numpy import ndarray, dtype
import torch.optim as optim
from collections import Counter
import data_generation
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
import graph_conv as gr
import torch
from collections import defaultdict

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
    
    def unary_discovery(self, interv_node:int, true_causal_graph: nx.DiGraph, pcdag:np.ndarray, data:pd.DataFrame)->np.ndarray:
        new_pcdag = pcdag.copy()
        interv_data = self.intervene(true_causal_graph, data, interv_node)
        neighbors = self.get_neighbors(pcdag, interv_node)

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
        #np.random.seed(seed = 47)
        #random.seed(47)
        G = nx.from_numpy_array(pcdag_matrix, create_using = nx.DiGraph)

        # adds i<j so that we only include each undirected edge once
        undirected_edges = [(i, j) for i, j in G.edges if G.has_edge(j, i) and i<j]
        random.shuffle(undirected_edges)

        for u, v in undirected_edges:
            if G.has_edge(v, u):
                G.remove_edge(v, u)
                G.remove_edge(u, v)

        for u, v in undirected_edges:
            if not G.has_edge(u, v) and not G.has_edge(v, u):
                choice = random.randint(1,2)
                if choice==1:
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

        G = nx.from_numpy_array(cpdag, create_using = nx.DiGraph)
        undirected_edges = [(i, j) for i, j in G.edges if G.has_edge(j, i) and i<j]
        for u, v in undirected_edges:
            G.remove_edge(v, u)
            G.remove_edge(u, v)


        original_num_isolated_nodes = nx.number_of_isolates(G)
        edges = list(G.edges)
        #print('edges')
        #print(len(edges))
        #print(original_num_isolated_nodes)
        subgraphs = []
        subset_size = np.random.randint(1,  len(edges) + 1)
        successful_draws = 0

        while successful_draws != subset_size:
            subset_size_graph = np.random.randint(1,  len(edges) + 1)
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

    def model_train_data(self, cpdag:np.ndarray):
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

        experiment = Experiment(5, 5)
        rand_subsam_matrices = experiment.rand_subsam_w_rep(cpdag)

        pc_matrices = [pc_a.pc(nx.from_numpy_array(matrix, create_using=nx.DiGraph)) for matrix in rand_subsam_matrices]

        return pc_matrices, rand_subsam_matrices

    def hamming_distance(self, pcdag:np.ndarray, true_causal_dag:np.ndarray):
        distance = np.sum((pcdag == 0) & (true_causal_dag == 1)) + np.sum((pcdag == 1) & (true_causal_dag == 0))
        return distance

    def random_design(self, pcdag:np.ndarray, true_causal_graph:nx.DiGraph, true_causal_dag:np.ndarray, data:pd.DataFrame, k:int):
        sampled_edge_indices = np.random.choice(self.get_num_nodes(pcdag), size=k, replace=False)
        hamming_distances = []
        num_interv_ran = 0
        updated_pcdag = pcdag.copy()
        for i, node in enumerate(sampled_edge_indices):
            if self.hamming_distance(updated_pcdag, true_causal_dag) == 0:
                hamming_distances.append(self.hamming_distance(updated_pcdag, true_causal_dag))
                num_interv_ran += 1
                break
            else:
                updated_pcdag = self.unary_discovery(interv_node=node, true_causal_graph=true_causal_graph, pcdag=updated_pcdag, data = data)
                hamming_distances.append(self.hamming_distance(updated_pcdag, true_causal_dag))
                num_interv_ran += 1
        return hamming_distances, num_interv_ran,sampled_edge_indices
    
    def get_unoriented_nodes(self, pcdag: np.ndarray) -> np.ndarray:
        unoriented_nodes = []
        for i in range(pcdag.shape[0]):
            if any(pcdag[i, :] == 1) and any(pcdag[:, i] == 1):  # Adjust condition based on edge encoding
                unoriented_nodes.append(i)
        return np.array(unoriented_nodes)

    def random_adv_design(self, pcdag:np.ndarray, true_causal_graph:nx.DiGraph, true_causal_dag:np.ndarray, data:pd.DataFrame, k:int):
        unoriented_nodes = self.get_unoriented_nodes(pcdag)
        sampled_edge_indices = np.random.choice(unoriented_nodes, size=min(k, len(unoriented_nodes)), replace=False)
        hamming_distances = []
        num_interv_ran = 0
        updated_pcdag = pcdag.copy()
        for i, node in enumerate(sampled_edge_indices):
            if self.hamming_distance(updated_pcdag, true_causal_dag) == 0:
                hamming_distances.append(self.hamming_distance(updated_pcdag, true_causal_dag))
                num_interv_ran += 1
                break
            else:
                updated_pcdag = self.unary_discovery(interv_node=node, true_causal_graph=true_causal_graph, pcdag=updated_pcdag, data = data)
                hamming_distances.append(self.hamming_distance(updated_pcdag, true_causal_dag))
                num_interv_ran += 1
        return hamming_distances, num_interv_ran, sampled_edge_indices
    
    def get_trainloader(self, pcdag:np.ndarray):
        trainloader = []
        pc_matrices, rand_subsam_matrices = self.model_train_data(cpdag = pcdag)
        for x, y in zip(pc_matrices, rand_subsam_matrices):
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=torch.float32, requires_grad=True)
            if not isinstance(y, torch.Tensor):
                y = torch.tensor(y, dtype=torch.float32)
            trainloader.append((x, y))
        return trainloader

    # helper for maximal_disagreement
    def calculate_entropy(self, orientations: List) -> float:
        # Use Counter to count occurrences of each orientation
        orientation_counts = Counter(orientations)

        # Calculate the proportions (p_i) for each orientation
        total_votes = len(orientations)
        entropy = 0.0 # returns 0 if "orientations" is empty
        for count in orientation_counts.values():
            p_i = count / total_votes
            entropy -= p_i * np.log2(p_i)  # Shannon entropy formula

        return entropy

    # helper for maximal_disagreement
    def preprocess_orientations(self, adj_matrices: List[np.ndarray]) -> dict:
        num_models = len(adj_matrices)
        matrix_dim = len(adj_matrices[0])

        edge_orientations = defaultdict(list)

        for m in range(num_models):
            for i in range(matrix_dim):
                for j in range(matrix_dim):
                    if i == j:
                        continue  # Skip self-loops
                    if adj_matrices[m][i][j] == 1.0 and adj_matrices[m][j][i] == 0.0:
                        edge_orientations[(i, j)].append(1)
                    elif adj_matrices[m][j][i] == 1.0 and adj_matrices[m][i][j] == 0.0:
                        edge_orientations[(i, j)].append(0)
                    else:
                        pass  # Handle as needed

        return edge_orientations

    # helper for maximal_disagreement
    def calculate_node_entropy(self, node, edge_orientations):
        total_entropy = 0.0
        for (i, j), orientations in edge_orientations.items():
            if i == node or j == node:  # Consider only edges incident to the node
                total_entropy += self.calculate_entropy(orientations)

        return total_entropy

    #predictions is list of pcdags returns node of most disagreement
    def get_maximal_disagreement(self, predictions:list[ndarray])->int:
        edge_orientations = self.preprocess_orientations(predictions)

        num_nodes = len(predictions[0])

        # Calculate entropy for each node
        node_entropies = np.zeros(num_nodes)
        for node in range(num_nodes):
            node_entropies[node] = self.calculate_node_entropy(node, edge_orientations)

        # Find the node with the maximum entropy
        max_entropy_node = np.argmax(node_entropies)
        return int(max_entropy_node)

    def majority_vote(self, predictions:List[ndarray])->np.ndarray:
        n_nodes = predictions[0].shape[0]
        final_pcdag = np.zeros((n_nodes, n_nodes), dtype=int)
        for i in range(n_nodes):
            for j in range(n_nodes):
                edge_types = [(pcdag[i, j], pcdag[j, i]) for pcdag in predictions]
                edge_count = Counter(edge_types)
                most_common_edge, _ = edge_count.most_common(1)[0]
                final_pcdag[i, j] = most_common_edge[0]
                final_pcdag[j, i] = most_common_edge[1]
        return final_pcdag

    """
    def qbc(self, epochs:int, committee_size:int, pcdag:np.ndarray, true_causal_dag:np.ndarray, true_causal_graph:nx.DiGraph, data:pd.DataFrame, k:int, _lambda:int):
        for interv in k:

            
            #somewhere in loop need to cjeck if pcdag is now dag and break loop if so
            

            committee = [GCN(len(pcdag)) for member in range(committee_size)]
            optimizers = [optim.Adam(model.parameters(), lr=0.01) for model in committee]
            trainloaders = [self.get_trainloader(pcdag = pcdag) for member in range(committee_size)]
            #for epoch in epochs:
            #   committee_results = None
            predictions = []
            for i, member in enumerate(committee):
                member.run_train(self, epochs, optimizers[i], trainloaders[i], _lambda)
                prediction = member.predict_pcdag(pcdag = pcdag)
                predictions.append(prediction)

            committee_pcdag = self.majority_vote(predictions = predictions)
            maximal_disagreed_node = self.get_maximal_disagreement([0])
            updated_pcdag = self.unary_discovery(interv_node = maximal_disagreed_node, true_causal_graph = true_causal_graph, pcdag = pcdag, data = data)
            pcdag = updated_pcdag
    """






        #for model in committee:
            #train all models and predict DAG
            #choose node with most disagreement (max entropy)
            #intervene on node
            #retrain models
            #rinse 

        #    pass

    def qbc(self, epochs:int, committee_size:int, pcdag:np.ndarray, true_causal_dag:np.ndarray, true_causal_graph:nx.DiGraph, data:pd.DataFrame, k:int, _lambda:int):
            hamming_distances = []
            num_interv_ran = 0
            pcdag = pcdag.copy()
            for interv in range(k):

                """
                somewhere in loop need to cjeck if pcdag is now dag and break loop if so
                """

                committee = [gr.GCN(len(pcdag)) for member in range(committee_size)]
                optimizers = [optim.Adam(model.parameters(), lr=0.01) for model in committee]
                trainloaders = [self.get_trainloader(pcdag = pcdag) for member in range(committee_size)]
                #for epoch in epochs:
                #   committee_results = None
                predictions = []

                for i, member in enumerate(committee):
                    member.run_train(epochs, optimizers[i], trainloaders[i], _lambda)
                    _, prediction = member.predict_pcdag(pcdag = pcdag)
                    predictions.append(prediction)

                # if committe DAG is right than break and no more interventions
                committee_pcdag = self.majority_vote(predictions = predictions)
                hamming_distances.append(self.hamming_distance(updated_pcdag, true_causal_dag=true_causal_dag))
                if committee_pcdag == true_causal_dag:
                    break
                #add statistics here 
                maximal_disagreed_node = self.get_maximal_disagreement(predictions = predictions)
                updated_pcdag = self.unary_discovery(interv_node = maximal_disagreed_node, true_causal_graph = true_causal_graph, pcdag = pcdag, data = data)
                #add statistics here 
                pcdag = updated_pcdag

                num_interv_ran += 1
            return hamming_distances, num_interv_ran


if __name__ == '__main__':
    np.random.seed(seed=47)
    random.seed(47)
    G = dg.create_dag(n=20, expected_degree=1)
    start_adj_matrix = nx.to_numpy_array(G)
    pcdag = pc_a.pc(G)

    experiment = Experiment(5, 5)
    shared_pos = experiment.visualize_pcdag(pcdag, title="PCDAG")
    
    true_DAG, DAG = experiment.random_dag_from_pcdag(pcdag) 
    hamming_distances, num_interv_ran = experiment.qbc(epochs = 1, committee_size = 3, pcdag = pcdag, true_causal_dag=true_DAG, true_causal_graph = DAG, data = dg.generate_data(DAG), k = 1, _lambda = 0.5)    #hamming, num, sampled_edge_indices = experiment.random_adv_design(pcdag = pcdag, true_causal_graph = DAG, true_causal_dag = true_DAG, data = dg.generate_data(graph = DAG), k = 10)
    #print(hamming)
    #print(num)
    #print(sampled_edge_indices)


"""

    sample_subgraphs = experiment.rand_subsam_w_rep(cpdag=pcdag)

    for subgraph in sample_subgraphs:
        #print(len(subgraph))
        print("subgraph: ")
        experiment.visualize_pcdag(subgraph, pos=shared_pos, title="true DAG")

        subgraph_pc = pc_a.pc(nx.from_numpy_array(subgraph, create_using = nx.DiGraph))

        print("subgraph pc: ")
        experiment.visualize_pcdag(subgraph_pc, pos=shared_pos, title="pc of subgraph")

np.random.seed(seed=47)
random.seed(47)
G = dg.create_dag(n=10, expected_degree=1)
start_adj_matrix = nx.to_numpy_array(G)
pcdag = pc_a.pc(G)
experiment = Experiment(5, 5)
shared_pos = experiment.visualize_pcdag(pcdag, title="PCDAG")
true_DAG, DAG = experiment.random_dag_from_pcdag(pcdag)  # gets random graph from MEC(s)


sample_subgraphs = experiment.rand_subsam_w_rep(cpdag=pcdag)

for subgraph in sample_subgraphs:
    #print(len(subgraph))
    print("subgraph: ")
    experiment.visualize_pcdag(subgraph, pos=shared_pos, title="true DAG")

    subgraph_pc = pc_a.pc(nx.from_numpy_array(subgraph, create_using = nx.DiGraph))

    print("subgraph pc: ")
    experiment.visualize_pcdag(subgraph_pc, pos=shared_pos, title="pc of subgraph")
    print(subgraph_pc)
"""