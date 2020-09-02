"""Defines the neural network, loss function and metrics"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.dynamic_reduction_network import DynamicReductionNetwork

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.drn = DynamicReductionNetwork(input_dim=4, hidden_dim=64,
                                           k = 8,
                                           output_dim=2, aggr='max',
                                           norm=torch.tensor([1./910.5, 1./6., 1./3.1416016, 1./1.0693359]))
    def forward(self, data):
        logits = self.drn(data)
        return logits

def loss_fn(prediction, truth):
    return F.mse_loss(prediction, truth, reduction='mean')
