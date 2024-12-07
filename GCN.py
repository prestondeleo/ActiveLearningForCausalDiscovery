import numpy as np
import pandas as pd
import math
import torch_geometric
import torch.nn as nn
import torch.optim as optim
import torch


class GCN(nn.Module):
    def __init__(self, vertices):
        super(GCN, self).__init__()
        self.gcn1 = None
        self.gcn2 = None
        self.fc1 = nn.Linear()        
        
    def forward(self, pcdag:np.ndarray):

        pass

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

    def train(self, epochs,pcdag:np.ndarray):
        pass
