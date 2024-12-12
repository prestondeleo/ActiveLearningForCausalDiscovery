import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch
from torch_geometric.nn import GCNConv
from torch_geometric.utils import dense_to_sparse
import active_learning as al
import random
import data_generation as dg
import pc_algorithm as pc_a
import networkx as nx
import torch.optim as optim

class GCN(nn.Module):
    def __init__(self, input_dim:int):
        super(GCN, self).__init__()
        self.gcn1 = GCNConv(input_dim, 16)
        self.gcn2 = GCNConv(16, 8)
        self.fc1 = nn.Linear(16,1)      
    
    def forward(self, pcdag: np.ndarray):
        adjacency_matrix = torch.tensor(pcdag, dtype=torch.float32)
        edge_index, _ = dense_to_sparse(adjacency_matrix)
        undirected_mask = (adjacency_matrix == 1) & (adjacency_matrix.T == 1)
        undirected_edges = torch.nonzero(undirected_mask.triu(), as_tuple=False) 
        node_features = torch.eye(len(pcdag), dtype=torch.float32)

        x = self.gcn1(node_features, edge_index)
        x = torch.relu(x)
        x = self.gcn2(x, edge_index)
        x = torch.relu(x)

        directed_adj = adjacency_matrix.clone().detach()

        if undirected_edges.size(0) > 0:
            edge_features = torch.cat([x[undirected_edges[:, 0]], x[undirected_edges[:, 1]]], dim=1)
            edge_orientations = torch.sigmoid(self.fc1(edge_features)).squeeze()

            num_nodes = adjacency_matrix.size(0)
            flat_directed_adj = directed_adj.view(-1)
            indices_for_srcdst = undirected_edges[:, 0] * num_nodes + undirected_edges[:, 1]
            indices_for_dstsrc = undirected_edges[:, 1] * num_nodes + undirected_edges[:, 0]

            flat_directed_adj = flat_directed_adj.scatter(0, indices_for_srcdst, edge_orientations)
            flat_directed_adj = flat_directed_adj.scatter(0, indices_for_dstsrc, 1 - edge_orientations)
            directed_adj = flat_directed_adj.view(num_nodes, num_nodes)

        return directed_adj

    def reconstruction_loss(self, predicted_dag:np.ndarray, true_dag:np.ndarray):
        loss = nn.functional.mse_loss(predicted_dag, true_dag)
        return loss

    def acyclic_loss(self, predicted_dag:np.ndarray):
        d = predicted_dag.size(0)
        exponentiation_matrix = torch.matrix_exp(predicted_dag * predicted_dag)
        trace = torch.trace(exponentiation_matrix)
        loss = trace - d
        return loss

    def full_loss(self, predicted_dag:np.ndarray, true_dag:np.ndarray, lambda_:float):
        loss = self.reconstruction_loss(predicted_dag, true_dag) + lambda_ * self.acyclic_loss(predicted_dag)
        return loss
    
    def run_train(self, epochs, optimizer, dataloader, _lambda):
        results = []
        for epoch in range(epochs):
            total_loss = 0
            pred = None
            for pcdag, true_dag in dataloader:
                print(f"epoch: {epoch + 1}, batch: {1}/{len(dataloader)}")
                optimizer.zero_grad()
                predicted_dag = model(pcdag)
                pred = predicted_dag
                pred = (pred > 0.5).int()
                results.append(pred)
                print(predicted_dag)
                true_dag_tensor = torch.tensor(true_dag, dtype=torch.float32) if not isinstance(true_dag, torch.Tensor) else true_dag
                predicted_dag_tensor = torch.tensor(predicted_dag, dtype=torch.float32) if not isinstance(predicted_dag, torch.Tensor) else predicted_dag
                loss = self.full_loss(predicted_dag_tensor, true_dag_tensor, _lambda)

                #loss = self.full_loss(edge_orientations, true_dag, _lambda)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                print(total_loss)

        #return pred
        return results
    


if __name__ == '__main__':
    """
    np.random.seed(seed = 47)  
    random.seed(47)
    G = dg.create_dag(n = 20, expected_degree = 1)
    start_adj_matrix = nx.to_numpy_array(G)        
    pcdag = pc_a.pc(G)
    experiment = al.Experiment(5, 5)
    true_DAG, DAG = experiment.random_dag_from_pcdag(pcdag) #gets random graph from MEC(s)
    model = GCN(input_dim=len(pcdag))
    oriented_adj = model(pcdag)
    trainloader = [(pcdag,true_DAG)]
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    pred = model.run_train(epochs = 2, optimizer = optimizer, dataloader=trainloader, _lambda = 0.5)
    """
    np.random.seed(seed = 47)  
    random.seed(47)
    G = dg.create_dag(n = 10, expected_degree = 1)
    start_adj_matrix = nx.to_numpy_array(G)        
    pcdag = pc_a.pc(G)
    experiment = al.Experiment(5, 5)
    shared_pos = experiment.visualize_pcdag(pcdag, title="PCDAG")
    true_DAG, DAG = experiment.random_dag_from_pcdag(pcdag) #gets random graph from MEC(s)

    experiment.visualize_pcdag(true_DAG, pos=shared_pos, title="true DAG")

    model = GCN(input_dim=len(pcdag))
    oriented_adj = model(pcdag)
    experiment.visualize_pcdag(oriented_adj, pos=shared_pos, title="GCN DAG")
    trainloader = [(pcdag,true_DAG)]
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    #model.train(epochs = 2, optimizer = optimizer, dataloader=trainloader, _lambda = 0.5)
    results = model.run_train(epochs = 10, optimizer = optimizer, dataloader=trainloader, _lambda = 0.5)
    for result in results:
        experiment.visualize_pcdag(result, pos=shared_pos, title="GCN DAG")





  #trainloader = [(pcdag,true_DAG)]
    #trainloader = [(torch.tensor(pcdag, dtype=torch.float32), torch.tensor(true_DAG, dtype=torch.float32))]
