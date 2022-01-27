import uproot
import argparse
import os.path as osp
import os

import numpy as np
import torch
from torch_geometric.data import DataLoader


import model.net as net
import utils


parser = argparse.ArgumentParser()
parser.add_argument('--data', default='data',
                    help="Name of the data folder")
parser.add_argument('--restore_file', default='best', help="name of the file in --model_dir \
                     containing weights to load")
parser.add_argument('--ckpts', default='ckpts',
                    help="Name of the ckpts folder")
parser.add_argument('--mode', default='fix',
                    help="simple for simple fixed size GNN, fix for fixed size GNN with DeepMET structure, dyn for dynamical GNN")
parser.add_argument('--out', default='',
                    help="additional name info if gnn is applied to other data")

# ======================================

"""Evaluates the model"""
"""
import logging
import time
from tqdm import tqdm

import json
from torch.autograd import Variable

import utils

from torch_geometric.utils import to_undirected
from torch_cluster import radius_graph, knn_graph

import matplotlib.pyplot as plt
import mplhep as hep
plt.style.use(hep.style.CMS)



def evaluate(model, device, loss_fn, dataloader, metrics, deltaR, deltaR_dz, model_dir, out='', mode="fix"):
    """ """
    Evaluate the model on `num_steps` batches.

    Args:
        model: (torch.nn.Module) the neural network
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        num_steps: (int) number of batches to train on, each of size params.batch_size
    """ """
    # set model to evaluation mode
    model.eval()

    with torch.no_grad():
        # summary for current eval loop

        # compute metrics over the dataset
        with tqdm(total=len(dataloader)) as t:
            for data in dataloader:
                
                data = data.to(device)

                phi = torch.atan2(data.x[:,1], data.x[:,0])
                etaphi = torch.cat([data.x[:,3][:,None], phi[:,None]], dim=1)
                edge_index = radius_graph(etaphi, r=deltaR, batch=data.batch, loop=True, max_num_neighbors=500)

                result = model(data.x, edge_index=edge_index, batch=data.batch)

                loss = loss_fn(result, data.x, data.y, data.batch)
                
                t.update()



#if __name__ == '__main__':
def main_func():
    """ """
        Evaluate the model on the test set.
    
    # Load the parameters
    args = parser.parse_args()

    # fetch dataloaders
    dataloaders = data_loader.fetch_dataloader(data_dir=osp.join(os.environ['PWD'],args.data), 
                                               batch_size=120, 
                                               validation_split=.2)
    test_dl = dataloaders['test']

    # Define the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = net.Net(7, 3, output_dim=1, hidden_dim=32, conv_depth=4, mode=args.mode).to(device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=500, threshold=0.05)

    loss_fn = net.loss_fn
    metrics = net.metrics
    model_dir = osp.join(os.environ['PWD'],args.ckpts)
    deltaR = 0.4
    deltaR_dz = 10


    # Reload weights from the saved file
    restore_ckpt = osp.join(model_dir, args.restore_file + '.pth.tar')
    ckpt = utils.load_checkpoint(restore_ckpt, model, optimizer, scheduler)
    epoch = ckpt['epoch']
    #utils.load_checkpoint(os.path.join(model_dir, args.restore_file + '.pth.tar'), model)
    with open(osp.join(model_dir, 'metrics_val_best.json')) as restore_metrics:
            best_validation_loss = json.load(restore_metrics)['loss']

    # Evaluate
    evaluate(model, device, loss_fn, test_dl, metrics, deltaR, deltaR_dz, model_dir, args.out)



# ======================================
"""
def load_pfinfo(tree, event):
    particle_list = np.column_stack([
        np.multiply(np.array(tree["PFCands_pt"].array()[event]), np.cos(np.array(tree["PFCands_phi"].array()[event]))),
        np.multiply(np.array(tree["PFCands_pt"].array()[event]), np.sin(np.array(tree["PFCands_phi"].array()[event]))),
        np.array(tree["PFCands_pt"].array()[event]),
        np.array(tree["PFCands_eta"].array()[event]),
        np.array(tree["PFCands_d0"].array()[event]),
        np.array(tree["PFCands_dz"].array()[event]),
        np.array(tree["PFCands_mass"].array()[event]),
        np.array(tree["PFCands_pdgId"].array()[event]),
        np.array(tree["PFCands_charge"].array()[event]),
        np.array(tree["PFCands_fromPV"].array()[event]),
    ])
    print(particle_list)
    return particle_list


if __name__=='__main__':
    args = parser.parse_args()

    pathName = "/nfs/dust/cms/user/jdriesch/monotopv2/smeacol/test/"
    tree = pathName + "mc_tree.root"
    """
    with uproot.open(tree) as file0:
        tree0 = file0["Events"]
        data0 = load_pfinfo(tree0, 0) # can be adjusted to get other events in tree
    """
    #data = DataLoader(data0, batch_size=1, shuffle=False)
    device = torch.device('cpu')
    print(device)

    model = net.Net(7, 3, output_dim=1, hidden_dim=32, conv_depth=4, mode=args.mode).to(device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=500, threshold=0.05)

    loss_fn = net.loss_fn
    metrics = net.metrics
    model_dir = osp.join(os.environ['PWD'],args.ckpts)
    deltaR = 0.4

    # Reload weights from the saved file
    restore_ckpt = osp.join(model_dir, args.restore_file + '.pth.tar')
    print(restore_ckpt)
    ckpt = utils.load_checkpoint(restore_ckpt, model, optimizer, scheduler)
    epoch = ckpt['epoch']
    #utils.load_checkpoint(os.path.join(model_dir, args.restore_file + '.pth.tar'), model)
    with open(osp.join(model_dir, 'metrics_val_best.json')) as restore_metrics:
            best_validation_loss = json.load(restore_metrics)['loss']    

