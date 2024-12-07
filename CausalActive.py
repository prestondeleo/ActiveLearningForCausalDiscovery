import data_generation as dg
import pc_algorithm as pc_a
import networkx as nx
import numpy as np
import pandas as pd
import random
import dowhy

from dowhy import CausalModel
"""
1. Need to generate synthetic graphs with real causal graphs

2. Query by committee active learning approach ()

3. Compare with other models (the two we talked about, others?)

4. Analysis

only do discovery on edge orientations and not on existence of edges or nodes

committe performs DAG generation beyon MEC (Markov Equivalence class)


"""





class causalQBC:
    def __init__(self, num_models:int, k:int)-> None:
        self.num_models = num_models
        self.k = k
        #self.sample_size = sample_size

    def get_num_nodes(self, cpdag:np.ndarray)->int:
        return len(cpdag)

    def rand_subsam_w_rep(self, cpdag:np.ndarray, num_nodes, min_perc_samp = 0.25, max_perc_samp = 0.9)->np.ndarray:
        #sample size would be number of nodes. 
        #could be called subset size
        num_nodes = self.get_num_nodes(cpdag)#adj_matric.shape[0]
        #subset_indices
        subset_size = np.random.randint(int(min_perc_samp * num_nodes),  int(max_perc_samp * num_nodes) + 1  )

        sampled_indices = np.random.choice(num_nodes, size=subset_size, replace=True)

        #subgraph_adjmatrix = np.ix_()
        #Does not gurantee all nodes will be connected to at least one other node (single vertex with no edge)
        subgraph_adjmatrix = cpdag.ix_(sampled_indices,sampled_indices)

        return subgraph_adjmatrix
    
    def model_dag():
        pass




    def qbc_query(pcdag:np.ndarray, dag:np.ndarray, k, ):



        """
        telmiggido sleeps while the man belows!
        """
        pass


    def query():
        pass

    #NEED TO DETERMINE IF CORRECT BELOW

    def get_neighbors(self, pcdag:np.ndarray, node:int)->np.ndarray:
        neighbors = np.where((pcdag[node, :] == 1) | (pcdag[:, node] == 1))[0]
        return neighbors
    
    ###get_neighbors seems to work


    def intervene_orient(self,pcdag:np.ndarray,data:pd.DataFrame, interv_node, neighbors):
        oriented_edges = []
        #treatment = data.columns[interv_node] 

        G = nx.DiGraph(pcdag)
        graph = nx.nx_pydot.to_pydot(G).to_string()
        data.columns = data.columns.astype(str)
        for neighbor in neighbors:

            outcome = str(data.columns[neighbor])
            treatment = str(data.columns[interv_node])
            model = CausalModel(
                            data=data,
                            treatment=treatment,
                            outcome=outcome,
                            graph=graph
            )
        identified_estimand = model.identify_effect()

        causal_estimate = model.estimate_effect(
            identified_estimand,
            method_name="backdoor.linear_regression" #I DONT KNOW IF THIS IS RIGHT!!!!
        )

        print(f"Causal Estimate of {interv_node} -> {neighbor}: {causal_estimate.value}")
        
        if causal_estimate.value != 0:  # Adjust threshold based on domain knowledge
            oriented_edges.append((interv_node, neighbor))  # Direct effect found
    
        return oriented_edges




    def random(self, pcdag:np.ndarray, data:pd.DataFrame, k:int):
        self.get_num_nodes(pcdag)
        nodes = list(range(self.get_num_nodes(pcdag)))
        interventional_nodes = random.sample(nodes, k)

        for interv_node in interventional_nodes:
            pass

        pass

    def randomadv_query():
        pass

if __name__ == '__main__':
    G = dg.create_dag(n = 20, expected_degree = 1)

    dg.display(G)

    adj_matrix = pc_a.pc(G)

    data = dg.generate_data(G)

    print(data)

    experiment = causalQBC(5, 5)
    print(experiment.get_neighbors(pcdag = adj_matrix,node = 1))

    print(experiment.intervene_orient(pcdag=adj_matrix,data = data, interv_node = 3, neighbors =experiment.get_neighbors(pcdag = adj_matrix,node = 3) ))




