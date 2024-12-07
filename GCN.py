import numpy as np
import pandas as pd
import math
import torch_geometric
import torch.nn as nn
import torch.optim as optim


class GCN(nn.Module):
    def __init__(self, vertices):
        super(GCN, self).__init__()
        self.gcn1 = None
        self.gcn2 = None
        self.fc1 = nn.Linear()        
        
    def forward(self, pcdag:np.ndarray):

        pass

    def reconstruction_loss(self):
        pass

    def acyclic_loss(self, lambda_):
        pass

    def full_loss(self):
        pass

    def train(self, epochs,pcdag:np.ndarray):
        pass
