import json
import os.path as osp
import os
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_undirected
from torch_cluster import radius_graph, knn_graph
from torch_geometric.datasets import MNISTSuperpixels
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from tqdm import tqdm
import argparse
import utils
import model.net as net
import model.data_loader as data_loader
import warnings

import matplotlib.pyplot as plt
import mplhep as hep
plt.style.use(hep.style.CMS)


warnings.simplefilter('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--data', default='data',
                    help="Name of the data folder")
parser.add_argument('--ckpts', default='ckpts',
                    help="Name of the ckpts folder")

def plot_input_features(data, plots=list(range(11))):
    name=["pX", "pY", "pT", "eta", "d0", "dz", "mass", "puppiWeight", "pdgId", "charge", "fromPV"]
    data0 = []
    print(len(data.x[:,0]))
    for j in range(len(data.x[:,0])):
        data0.append(np.arctan(data.x[j,1].cpu().numpy()*1./data.x[j,0].cpu().numpy()))
    #print(data0)
    plt.hist(data0, 100)
    #plt.yscale('log')
    #plt.show()
    plt.savefig("znunu_inputs/phi_1.pdf")
    plt.clf
    print("done")
    """
    for i in plots:
        data0=data.x[:,i].cpu().numpy()
        print(data0)
        nbins=50
        if i==8: nbins=450
        plt.hist(data0, nbins)
        plt.yscale('log')
        plt.show()
        plt.savefig("znunu_inputs/"+name[i]+"_1.pdf")
        plt.clf()
    """

def plot_pdg_dependent(data):
    data0=data.x
    #pdg=data.x[:,8].numpy()
    pdgs=[1, 2, 11, 13, 22, 130, 211]
    for pdg in pdgs:
        mass = data0[abs(data0[:,8])==pdg][:,6]
        plt.hist(mass.numpy(), 50)
        plt.yscale('log')
        plt.show()
        #plt.savefig("znunu_inputs/pdg_mass/"+str(pdg)+".pdf")
        #plt.savefig("znunu_inputs/pdg_mass/"+str(pdg)+".png")
        plt.clf()
        print(pdg, " done")

def plot_next_neighbors(data, index):
    deltaR=0.8

    phi = torch.atan2(data.x[:,1], data.x[:,0])
    etaphi = torch.cat([data.x[:,3][:,None], phi[:,None]], dim=1) 
    edge_index = radius_graph(etaphi, r=deltaR, batch=data.batch, loop=True, max_num_neighbors=1000)
    edge_index_dim1 = edge_index[1,:] 
    # edge_index_dim1 = torch.cat((edge_index[0,:], edge_index[1,:]),0)
    num_nodes = max(edge_index_dim1).item()+1
    edge_count = torch.histc(edge_index_dim1, bins=num_nodes, min=0, max=num_nodes-1)
    max_num_neighbors = max(edge_count).item()
    #print("max: num neighbors:", max(edge_count).item())
    #n, bins, batches = plt.hist(edge_count.to('cpu').numpy())
    #plt.show()
    print("done: ",  index, "; max_num_neighbors:", max_num_neighbors)
    return num_nodes, max_num_neighbors

def plot_loss_func(epochs, train, eval):
        plt.plot(self.epoch, self.train_loss, label="training loss")
        plt.plot(self.epoch, self.eval_loss, label="evaluation loss")
        plt.legend()
        plt.savefig(checkpoint + "/loss.pdf")

if __name__ == '__main__':
    args = parser.parse_args()

    dataloaders = data_loader.fetch_dataloader(data_dir=osp.join(os.environ['PWD'],args.data), 
                                               batch_size=200,
                                               validation_split=0)
    train_dl = dataloaders['train']
    test_dl = dataloaders['test']

    print(len(train_dl), len(test_dl))
    hist_max_num_neighbors = []
    hist_num_nodes = []
    index=0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    torch.cuda.set_per_process_memory_fraction(0.5)
    torch.set_num_threads(2)

    with tqdm(total=len(train_dl)) as t:
        for data in train_dl:
            data.to(device)
            plot_input_features(data)
            # plot_pdg_dependent(data)
            num_nodes, max_num_neighbors = plot_next_neighbors(data, index)
            hist_max_num_neighbors.append(max_num_neighbors)
            hist_num_nodes.append(num_nodes)
            
            index+=1
            if index>100: break
    
    plt.hist(hist_max_num_neighbors, bins = 50)
    plt.xlabel('Max num of connected PF candidates')
    plt.ylabel('Number of events')
    plt.title('$\Delta R < 0.8$;  max: {}'.format(max(hist_max_num_neighbors)))
    plt.savefig('znunu_inputs/max_num_neighbors_deltar08.pdf')
    plt.clf()
    '''
    plt.hist(hist_num_nodes, bins = 50)
    plt.xlabel('Number of PF candidates per event')
    plt.ylabel('Number of events')
    plt.title('$\Delta R < 0.4$')
    plt.savefig('znunu_inputs/num_nodes_deltar08.pdf')
    plt.clf()
    '''
    corr = np.corrcoef(hist_num_nodes, hist_max_num_neighbors)
    print(corr)

    plt.plot(hist_num_nodes, hist_max_num_neighbors, ".")
    plt.xlabel('Number of PF candidates per event')
    plt.ylabel('Max num of connected PF candidates')
    plt.title('$\Delta R < 0.8$; correlation: {}'.format(round(corr[0][1], 3)))
    plt.savefig('znunu_inputs/correlation_nodes_deltar08.pdf')
