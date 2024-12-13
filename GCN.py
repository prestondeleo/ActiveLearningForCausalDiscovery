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

    def forward(self, pcdag):
        adjacency_matrix = (
            pcdag if isinstance(pcdag, torch.Tensor) else torch.tensor(pcdag, dtype=torch.float32, requires_grad=True)
        )
        edge_index, _ = dense_to_sparse(adjacency_matrix)

        undirected_mask = (adjacency_matrix == 1) & (adjacency_matrix.T == 1)
        undirected_edges = torch.nonzero(undirected_mask.triu(), as_tuple=False)
        node_features = torch.eye(len(pcdag), dtype=torch.float32, requires_grad=True)
        x = self.gcn1(node_features, edge_index)
        x = torch.relu(x)
        x = self.gcn2(x, edge_index)
        x = torch.relu(x)

        directed_adj = adjacency_matrix.clone()

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
        for epoch in range(epochs):
            total_loss = 0
            for i, (pcdag, true_dag) in enumerate(dataloader):
                optimizer.zero_grad()
                predicted_dag = self.forward(pcdag)
                loss = self.full_loss(predicted_dag, true_dag, _lambda)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            print(f"Epoch {epoch + 1}, Total Loss: {total_loss}")

    def predict_pcdag(self, pcdag:np.ndarray)->np.ndarray:
        predicted_dag = self.forward(pcdag)
        return predicted_dag, (predicted_dag > 0.5).int()

if __name__ == '__main__':
    np.random.seed(seed = 47)  
    random.seed(47)
    G = dg.create_dag(n = 10, expected_degree = 2)
    start_adj_matrix = nx.to_numpy_array(G)        
    pcdag = pc_a.pc(G)
    experiment = al.Experiment(5, 5)
    shared_pos = experiment.visualize_pcdag(pcdag, title="PCDAG")
    true_DAG, DAG = experiment.random_dag_from_pcdag(pcdag) #gets random graph from MEC(s)
    experiment.visualize_pcdag(true_DAG, pos=shared_pos, title="true DAG")
    model = GCN(input_dim=len(pcdag))
    trainloader = []
    pc_matrices, rand_subsam_matrices = experiment.model_train_data(cpdag = pcdag)
    for x, y in zip(pc_matrices, rand_subsam_matrices):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32, requires_grad=True)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.float32)
        trainloader.append((x, y))
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    results = model.run_train(epochs = 10, optimizer = optimizer, dataloader=trainloader, _lambda = 0.5)
    predicted_dag, useful_predicted_dag = model.predict_pcdag(pcdag)
    #print((predicted_dag))
    #print((predicted_dag > 0.5).int())
    #experiment.visualize_pcdag((predicted_dag > 0.5).int(), pos=shared_pos, title="predicted DAG")

