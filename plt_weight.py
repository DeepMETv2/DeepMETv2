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
from evaluate import evaluate
import warnings
warnings.simplefilter('ignore')

import logging
from torch.autograd import Variable
import matplotlib.pyplot as plt
import mplhep as hep
plt.style.use(hep.style.CMS)


parser = argparse.ArgumentParser()
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before \
                    training")  # 'best' or 'train'
parser.add_argument('--data', default='data',
                    help="Name of the data folder")
parser.add_argument('--ckpts', default='ckpts',
                    help="Name of the ckpts folder")
parser.add_argument('--save_plot', default=False,
                    help="save_plot while training")

def plot_weight(model, loss_fn, dataloader, metrics, deltaR, model_dir, saveplot):
    """plot_weight
    Args:
        model: (torch.nn.Module) the neural network
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        num_steps: (int) number of batches to train on, each of size params.batch_size
    """
    #model.eval()
    # summary for current eval loop
    qT_arr = []
    weight_arr = [1, 11, 13, 22, 130, 211]
    colors = {
        1: 'black',
        2: 'black',
        11: 'blue',
        13: 'green',
        22:  'magenta',
        130: 'red',
        211: 'orange',
    }
    labels = {
        1: 'HF Candidate',
        2: 'HF Candidate',
        11: 'Electron',
        13: 'Muon',
        22: 'Gamma',
        130: 'Neutral Hadron',
        211: 'Charged Hadron',
    }
    tot = torch.empty(0,3).to('cuda')
    # compute metrics over the dataset
    for data in dataloader:
        data = data.to('cuda')
        x_cont = data.x[:,:8]
        x_cat = data.x[:,8:].long()
        print("x_cont shape:", (x_cont.shape))
        print("x_cat shape:", (x_cat.shape))
        phi = torch.atan2(data.x[:,1], data.x[:,0])
        etaphi = torch.cat([data.x[:,3][:,None], phi[:,None]], dim=1)
        # NB: there is a problem right now for comparing hits at the +/- pi boundary
        edge_index = radius_graph(etaphi, r=deltaR, batch=data.batch, loop=True, max_num_neighbors=255)
        # compute model output
        result = model(x_cont, x_cat, edge_index, data.batch)
        print("result ",result.shape)
        print("tot ",tot.shape)
        #pX,pY,pT,eta,d0,dz,mass,puppiWeight,pdgId,charge,fromPV
        result = torch.stack((torch.abs(x_cat[:, 0]),torch.abs(x_cont[:, 7]),result),dim=1)
        tot = torch.cat((tot,result),0)
    minx=-0.15
    maxx=1.25
    binwidth=0.05
    bin_edges=[-0.05, 0.05, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 1.1] #puppi
    #bin_edges = np.arange(minx,maxx,binwidth)  #eta
    qT_hist=[]
    for i in range(1, len(bin_edges)):
        qT_hist.append((bin_edges[i]+bin_edges[i-1])/2.)
    
    weight_hists={}
    plt.figure()
    for key in (1, 22, 130): #weight_arr:
        print(labels[key])
        if (key==1):
            W_arr=tot[np.where( (tot[:,0].cpu() == key) | (tot[:,0].cpu() == 2) )].cpu().detach().numpy()
        else:
            W_arr=tot[np.where(tot[:,0].cpu() == key)].cpu().detach().numpy()
        #np.savez('test_'+labels[key]+'.npz', W_arr)
        #print(W_arr[:,])
        print("w_arr shape:", (W_arr[np.where( (W_arr[:,1]>1))][:,1]))
        weight_hist=[]
        weightE_hist=[]
        for i in range(1, len(bin_edges)):
            print("range:", bin_edges[i-1], bin_edges[i])
            W_i=W_arr[np.where( (W_arr[:,1]>=bin_edges[i-1]) & (W_arr[:,1]<bin_edges[i]) )][:,2]
            if(len(W_i)>10):
                weight_hist.append(np.mean(W_i))
                weightE_hist.append(np.std(W_i))
            else:
                weight_hist.append(0)
                weightE_hist.append(0)
        plt.figure(1)
        #plt.errorbar(qT_hist, weight_hist,yerr=weightE_hist, color=colors[key], label=labels[key]) with error bar
        plt.plot(qT_hist, weight_hist,  color=colors[key], label=labels[key])


    plt.figure(1)
    plt.axis([minx,maxx, 0, 1.2])
    #plt.xlabel(r'PF $p_{T}$ [GeV]')  
    #plt.xlabel(r'PF $|\eta|$ ')
    plt.xlabel(r'PUPPI Weight')
    plt.ylabel(r'GraphMet Weight')
    plt.legend()
    plt.xlim([0,1]) 
    #plt.xscale('log') # log scale
    plt.savefig('./Graph_weight_puppi.png')
    plt.clf()
    plt.close()

    return result


if __name__ == '__main__':
    args = parser.parse_args()

    dataloaders = data_loader.fetch_dataloader(data_dir=osp.join(os.environ['PWD'],args.data), 
                                               batch_size=60, 
                                               validation_split=0.3)
    train_dl = dataloaders['train']
    test_dl = dataloaders['test']

    print(len(train_dl), len(test_dl))
    
    #model = net.Net().to('cuda')
    #model = torch.jit.script(net.Net(7, 3)).to('cuda') # [px, py, pt, eta, d0, dz, mass], [pdgid, charge, fromPV]
    model = net.Net(8, 3).to('cuda')
    #optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-3)
    optimizer = torch.optim.AdamW(model.parameters(),lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=500, threshold=0.05)
    first_epoch = 0
    best_validation_loss = 10e7
    deltaR = 0.4

    loss_fn = net.loss_fn
    metrics = net.metrics

    model_dir = osp.join(os.environ['PWD'],args.ckpts)

    # reload weights from restore_file if specified
    if args.restore_file is not None:
        restore_ckpt = osp.join(model_dir, args.restore_file + '.pth.tar')
        ckpt = utils.load_checkpoint(restore_ckpt, model, optimizer, scheduler)
        first_epoch = ckpt['epoch']
        print('Restarting training from epoch',first_epoch)
        with open(osp.join(model_dir, 'metrics_val_best.json')) as restore_metrics:
            best_validation_loss = json.load(restore_metrics)['loss']

    for epoch in range(first_epoch+1,first_epoch+2):

        print('Current best loss:', best_validation_loss)
        if '_last_lr' in scheduler.state_dict():
            print('Learning rate:', scheduler.state_dict()['_last_lr'][0])

        plot_weight(model, loss_fn, test_dl, metrics, deltaR, model_dir, args.save_plot)


