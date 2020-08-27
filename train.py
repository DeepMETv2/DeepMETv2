import os.path as osp
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import MNISTSuperpixels
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from tqdm import tqdm_notebook as tqdm

from models.dynamic_reduction_network import DynamicReductionNetwork
from met_dataset import METDataset
from torch.utils.data.sampler import SubsetRandomSampler

import warnings
warnings.simplefilter('ignore')

transform = T.Cartesian(cat=False)
dataset = METDataset(os.environ['PWD']+'/data/')
dataset_size = len(dataset)
indices = list(range(dataset_size))
validation_split = .1
split = int(np.floor(validation_split * dataset_size))
random_seed= 42
np.random.seed(random_seed)
np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)
batch_size=60
train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
test_loader = DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler)
print('features ->', dataset.num_features)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.drn = DynamicReductionNetwork(input_dim=4, hidden_dim=64,
                                           k = 8,
                                           output_dim=2, aggr='max',
                                           norm=torch.tensor([1./910.5, 1./6., 1./3.1416016, 1./1.0693359]))
        
    def forward(self, data):
        logits = self.drn(data)
        return F.log_softmax(logits, dim=1)


model = Net().to('cuda')
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-3)

def train(epoch):
    model.train()
    for data in tqdm(train_loader):
        data = data.to('cuda')        
        optimizer.zero_grad()
        result = model(data, data.batch)
        mse = F.mse_loss(result, data.y, reduction='mean')
        mse.backward()
        optimizer.step()


def test():
    model.eval()
    correct = 0
    for data in test_loader:
        data = data.to('cuda:0')
        pred = model(data).max(1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(val_indices)
