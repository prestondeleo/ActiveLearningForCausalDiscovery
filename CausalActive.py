import data_generation as dg
import pc_algorithm as pc_a
import networkx as nx
import numpy as np
"""
1. Need to generate synthetic graphs with real causal graphs

2. Query by committee active learning approach ()

3. Compare with other models (the two we talked about, others?)

4. Analysis

only do discovery on edge orientations and not on existence of edges or nodes

committe performs DAG generation beyon MEC (Markov Equivalence class)


"""


G = dg.create_dag(n = 20, expected_degree = 1)

dg.display(G)

adj_matrx = pc_a.pc(G)


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




    def qbc_query():
        pass


    def query():
        pass

def random(causal_dag, data, num_interventions):
    pass
    #ground_truth = 




def randomadv_query():
    pass

