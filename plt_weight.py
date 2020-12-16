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
from torch_scatter import scatter_add

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
    label = {
        1: 'HF Candidate',
        11: 'Electron',
        13: 'Muon',
        22: 'Gamma',
        130: 'Neutral Hadron',
        211: 'Charged Hadron',
    }
    binedges_list = {
        'Pt': np.arange(-0.05,25.05,0.1),
        'eta': np.arange(-0.1,5.1,0.2),
        'Puppi': [-0.05, 0.05, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 1.1],
        'graph_weight': np.arange(-0.05,1.15,0.01),
        'qT1D': np.arange(0,420,20),
    }
    #print("bin list:", binedges_list)
    weight_pt_hist={}
    weight_eta_hist={}
    weight_puppi_hist={}
    weight_pt_histN={}
    weight_eta_histN={}
    weight_puppi_histN={}
    weight_CH_hist={}
    weight_qT_hist={}
    result = torch.empty(0,6).to('cuda')
    for key in weight_arr:
        weight_pt_hist[label[key]]=[]
        weight_pt_histN[label[key]]=[]
        for i in range(1, len(binedges_list['Pt'])):
            weight_pt_hist[label[key]].append(0)
            weight_pt_histN[label[key]].append(0)
        weight_eta_hist[label[key]]=[]
        weight_eta_histN[label[key]]=[]
        for i in range(1, len(binedges_list['eta'])):
            weight_eta_hist[label[key]].append(0)
            weight_eta_histN[label[key]].append(0)
    for key in (1,22,130):
        weight_puppi_hist[label[key]]=[]
        weight_puppi_histN[label[key]]=[]
        for i in range(1, len(binedges_list['Puppi'])):
            weight_puppi_hist[label[key]].append(0)
            weight_puppi_histN[label[key]].append(0)
    weight_CH_hist['puppi0']=[]
    weight_CH_hist['puppi1']=[]
    for i in range(1, len(binedges_list['graph_weight'])):
        weight_CH_hist['puppi0'].append(0)
        weight_CH_hist['puppi1'].append(0)
    weight_qT_hist['TrueMET']=[]
    weight_qT_hist['GraphMET']=[]
    weight_qT_hist['PFMET']=[]
    weight_qT_hist['PUPPIMET']=[]
    weight_qT_hist['DeepMETResponse']=[]
    weight_qT_hist['DeepMETResolution']=[]
    for i in range(1, len(binedges_list['qT1D'])):
        weight_qT_hist['TrueMET'].append(0)
        weight_qT_hist['GraphMET'].append(0)
        weight_qT_hist['PFMET'].append(0)
        weight_qT_hist['PUPPIMET'].append(0)
        weight_qT_hist['DeepMETResponse'].append(0)
        weight_qT_hist['DeepMETResolution'].append(0)

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
        TrueqT = torch.sqrt(data.y[:,0]**2+data.y[:,1]**2).cpu().detach().numpy()
        pfqT = torch.sqrt(data.y[:,2]**2+data.y[:,3]**2).cpu().detach().numpy()
        puppiqT = torch.sqrt(data.y[:,4]**2+data.y[:,5]**2).cpu().detach().numpy()
        deepMETResponseqT = torch.sqrt(data.y[:,6]**2+data.y[:,7]**2).cpu().detach().numpy()
        deepMETResolutionqT = torch.sqrt(data.y[:,8]**2+data.y[:,9]**2).cpu().detach().numpy()
        graphMETx = scatter_add(result*data.x[:,0], data.batch)
        graphMETy = scatter_add(result*data.x[:,1], data.batch)
        graphMETqT = torch.sqrt(graphMETx**2+graphMETy**2).cpu().detach().numpy()
        # qT 1D distribution
        for i in range(1, len(binedges_list['qT1D'])):
            binnedqT = TrueqT[np.where((TrueqT>=binedges_list['qT1D'][i-1]) & (TrueqT<binedges_list['qT1D'][i]) )]
            weight_qT_hist['TrueMET'][i-1]+=len(binnedqT)
            binnedqT = graphMETqT[np.where((graphMETqT>=binedges_list['qT1D'][i-1]) & (graphMETqT<binedges_list['qT1D'][i]) )]
            weight_qT_hist['GraphMET'][i-1]+=len(binnedqT)
            binnedqT = pfqT[np.where((pfqT>=binedges_list['qT1D'][i-1]) & (pfqT<binedges_list['qT1D'][i]) )]
            weight_qT_hist['PFMET'][i-1]+=len(binnedqT)
            binnedqT = puppiqT[np.where((puppiqT>=binedges_list['qT1D'][i-1]) & (puppiqT<binedges_list['qT1D'][i]) )]
            weight_qT_hist['PUPPIMET'][i-1]+=len(binnedqT)
            binnedqT = deepMETResponseqT[np.where((deepMETResponseqT>=binedges_list['qT1D'][i-1]) & (deepMETResponseqT<binedges_list['qT1D'][i]) )]
            weight_qT_hist['DeepMETResponse'][i-1]+=len(binnedqT)
            binnedqT = deepMETResolutionqT[np.where((deepMETResolutionqT>=binedges_list['qT1D'][i-1]) & (deepMETResolutionqT<binedges_list['qT1D'][i]) )]
            weight_qT_hist['DeepMETResolution'][i-1]+=len(binnedqT)

        #pX,pY,pT,eta,d0,dz,mass,puppiWeight,pdgId,charge,fromPV
        #pdg, pt, eta, puppi, weight
        ZQt = torch.gather(torch.sqrt(data.y[:,0]**2+data.y[:,1]**2), 0, data.batch)
        result = torch.stack((torch.abs(x_cat[:, 0]),torch.abs(x_cont[:, 2]),torch.abs(x_cont[:, 3]), torch.abs(x_cont[:, 7]), result,ZQt),dim=1)
        #result = result[np.where(result[:,5].cpu()<30 )]
        # weight vs pt
        # weight vs eta
        for key in weight_arr:
            if (key==1):
                W_arr=result[np.where( (result[:,0].cpu() == key) | (result[:,0].cpu() == 2) )].cpu().detach().numpy()
            else:
                W_arr=result[np.where(result[:,0].cpu() == key)].cpu().detach().numpy()
            for i in range(1, len(binedges_list['Pt'])):
                W_i=W_arr[np.where( (W_arr[:,1]>=binedges_list['Pt'][i-1]) & (W_arr[:,1]<binedges_list['Pt'][i]) )][:,4]
                weight_pt_hist[label[key]][i-1]+=np.sum(W_i)
                weight_pt_histN[label[key]][i-1]+=len(W_i)
            for i in range(1, len(binedges_list['eta'])):
                W_i=W_arr[np.where( (W_arr[:,2]>=binedges_list['eta'][i-1]) & (W_arr[:,2]<binedges_list['eta'][i]) )][:,4]
                weight_eta_hist[label[key]][i-1]+=np.sum(W_i)
                weight_eta_histN[label[key]][i-1]+=len(W_i)
        # weight vs puppi
        for key in (1,22,130):
            if (key==1):
                W_arr=result[np.where( (result[:,0].cpu() == key) | (result[:,0].cpu() == 2) )].cpu().detach().numpy()
            else:
                W_arr=result[np.where(result[:,0].cpu() == key)].cpu().detach().numpy()
            for i in range(1, len(binedges_list['Puppi'])):
                W_i=W_arr[np.where( (W_arr[:,3]>=binedges_list['Puppi'][i-1]) & (W_arr[:,3]<binedges_list['Puppi'][i]) )][:,4]
                weight_puppi_hist[label[key]][i-1]+=np.sum(W_i)
                weight_puppi_histN[label[key]][i-1]+=len(W_i)
        # weight distribution
        W_arr=result[np.where(result[:,0].cpu() == 211)].cpu().detach().numpy()
        for i in range(1, len(binedges_list['graph_weight'])):
            W_i=W_arr[np.where((W_arr[:,3]==0) & (W_arr[:,4]>=binedges_list['graph_weight'][i-1]) & (W_arr[:,4]<binedges_list['graph_weight'][i]) )][:,4]
            weight_CH_hist['puppi0'][i-1]+=len(W_i)
            W_i=W_arr[np.where((W_arr[:,3]==1) & (W_arr[:,4]>=binedges_list['graph_weight'][i-1]) & (W_arr[:,4]<binedges_list['graph_weight'][i]) )][:,4]
            weight_CH_hist['puppi1'][i-1]+=len(W_i)

    for key in weight_arr:
        for i in range(1, len(binedges_list['Pt'])):
            weight_pt_hist[label[key]][i-1]/=1.0*weight_pt_histN[label[key]][i-1]
            weight_pt_hist[label[key]] = np.nan_to_num(weight_pt_hist[label[key]])
        for i in range(1, len(binedges_list['eta'])):
            weight_eta_hist[label[key]][i-1]/=1.0*weight_eta_histN[label[key]][i-1]
            weight_eta_hist[label[key]] = np.nan_to_num(weight_eta_hist[label[key]])
    for key in (1,22,130):
        for i in range(1, len(binedges_list['Puppi'])):
            weight_puppi_hist[label[key]][i-1]/=1.0*weight_puppi_histN[label[key]][i-1]
            weight_puppi_hist[label[key]] = np.nan_to_num(weight_puppi_hist[label[key]])
    weights={
      'bin_edges':binedges_list,
      'weight_pt_hist':weight_pt_hist,
      'weight_eta_hist':weight_eta_hist,
      'weight_puppi_hist':weight_puppi_hist,
      'weight_CH_hist':weight_CH_hist,
      'weight_qT_hist':weight_qT_hist,
    }
    utils.save(weights, 'weight.plt')
    return result


if __name__ == '__main__':
    args = parser.parse_args()

    dataloaders = data_loader.fetch_dataloader(data_dir=osp.join(os.environ['PWD'],args.data), 
                                               batch_size=60, 
                                               validation_split=0.5)
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


