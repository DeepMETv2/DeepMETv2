"""Evaluates the model"""

import argparse
import logging
import os.path as osp
import os
import time

import numpy as np
import json
import torch
from torch.autograd import Variable

import utils
import model.net as net
import model.data_loader as data_loader

from torch_geometric.utils import to_undirected
from torch_cluster import radius_graph, knn_graph

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--restore_file', default='best', help="name of the file in --model_dir \
                     containing weights to load")
parser.add_argument('--data', default='data',
                    help="Name of the data folder")
parser.add_argument('--ckpts', default='ckpts',
                    help="Name of the ckpts folder")

def evaluate(model, device, loss_fn, dataloader, metrics, deltaR, deltaR_dz, model_dir):
    """Evaluate the model on `num_steps` batches.

    Args:
        model: (torch.nn.Module) the neural network
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        num_steps: (int) number of batches to train on, each of size params.batch_size
    """
    # set model to evaluation mode
    model.eval()

    # summary for current eval loop
    loss_avg_arr = []
    qT_arr = []
    has_deepmet = False
    resolutions_arr = {
        'MET':      [[],[],[]],
        'pfMET':    [[],[],[]],
        'puppiMET': [[],[],[]],
    }

    colors = {
        'pfMET': 'black',
        'puppiMET': 'red',
        'deepMETResponse': 'blue',
        'deepMETResolution': 'green',
        'MET':  'magenta',
    }

    labels = {
        'pfMET': 'PF MET',
        'puppiMET': 'PUPPI MET',
        'deepMETResponse': 'DeepMETResponse',
        'deepMETResolution': 'DeepMETResolution',
        'MET': 'DeepMETv2'
    }

    # compute metrics over the dataset
    for data in dataloader:

        has_deepmet = (data.y.size()[1] > 6)
        
        if has_deepmet == True and 'deepMETResponse' not in resolutions_arr.keys():
            resolutions_arr.update({
                'deepMETResponse': [[],[],[]],
                'deepMETResolution': [[],[],[]]
            })
        
        data = data.to(device)
        #x_cont = data.x[:,:7] #remove puppi
        x_cont = data.x[:,:8] #include puppi
        x_cat = data.x[:,8:].long()
        phi = torch.atan2(data.x[:,1], data.x[:,0])
        etaphi = torch.cat([data.x[:,3][:,None], phi[:,None]], dim=1)
        # NB: there is a problem right now for comparing hits at the +/- pi boundary 
        edge_index = radius_graph(etaphi, r=deltaR, batch=data.batch, loop=True, max_num_neighbors=255)
        result = model(x_cont, x_cat, edge_index, data.batch)

        #add dz connection
        #tic = time.time()
        #tinf = (torch.ones(len(data.x[:,5]))*float("Inf")).to('cuda')
        #edge_index_dz = radius_graph(torch.where(data.x[:,7]!=0, data.x[:,5], tinf), r=deltaR_dz, batch=data.batch, loop=True, max_num_neighbors=127)
        #cat_edges = torch.cat([edge_index,edge_index_dz],dim=1)
        #result = model(x_cont, x_cat, cat_edges, data.batch)
        #toc = time.time()
        #print('Event processing speed', toc - tic)

        loss = loss_fn(result, data.x, data.y, data.batch)

        # compute all metrics on this batch
        resolutions, qT= metrics['resolution'](result, data.x, data.y, data.batch)
        for key in resolutions_arr:
            for i in range(len(resolutions_arr[key])):
                resolutions_arr[key][i]=np.concatenate((resolutions_arr[key][i],resolutions[key][i]))
        qT_arr=np.concatenate((qT_arr,qT))
        loss_avg_arr.append(loss.item())

    # compute mean of all metrics in summary
    max_x=400 # max qT value
    x_n=40 #number of bins

    bin_edges=np.arange(0, max_x, 10)
    inds=np.digitize(qT_arr,bin_edges)
    qT_hist=[]
    for i in range(1, len(bin_edges)):
        qT_hist.append((bin_edges[i]+bin_edges[i-1])/2.)
    
    resolution_hists={}
    for key in resolutions_arr:

        R_arr=resolutions_arr[key][2] 
        u_perp_arr=resolutions_arr[key][0]
        u_par_arr=resolutions_arr[key][1]

        u_perp_hist=[]
        u_perp_scaled_hist=[]
        u_par_hist=[]
        u_par_scaled_hist=[]
        R_hist=[]

        for i in range(1, len(bin_edges)):
            R_i=R_arr[np.where(inds==i)[0]]
            R_hist.append(np.mean(R_i))
            u_perp_i=u_perp_arr[np.where(inds==i)[0]]
            u_perp_scaled_i=u_perp_i/np.mean(R_i)
            u_perp_hist.append((np.quantile(u_perp_i,0.84)-np.quantile(u_perp_i,0.16))/2.)
            u_perp_scaled_hist.append((np.quantile(u_perp_scaled_i,0.84)-np.quantile(u_perp_scaled_i,0.16))/2.)
            u_par_i=u_par_arr[np.where(inds==i)[0]]
            u_par_scaled_i=u_par_i/np.mean(R_i)
            u_par_hist.append((np.quantile(u_par_i,0.84)-np.quantile(u_par_i,0.16))/2.)
            u_par_scaled_hist.append((np.quantile(u_par_scaled_i,0.84)-np.quantile(u_par_scaled_i,0.16))/2.)

        u_perp_resolution=np.histogram(qT_hist, bins=x_n, range=(0,max_x), weights=u_perp_hist)
        u_perp_scaled_resolution=np.histogram(qT_hist, bins=x_n, range=(0,max_x), weights=u_perp_scaled_hist)
        u_par_resolution=np.histogram(qT_hist, bins=x_n, range=(0,max_x), weights=u_par_hist)
        u_par_scaled_resolution=np.histogram(qT_hist, bins=x_n, range=(0,max_x), weights=u_par_scaled_hist)
        R=np.histogram(qT_hist, bins=x_n, range=(0,max_x), weights=R_hist)
        resolution_hists[key] = {
            'u_perp_resolution': u_perp_resolution,
            'u_perp_scaled_resolution': u_perp_scaled_resolution,
            'u_par_resolution': u_par_resolution,
            'u_par_scaled_resolution':u_par_scaled_resolution,
            'R': R
        }
    metrics_mean = {
        'loss': np.mean(loss_avg_arr),
        #'resolution': (np.quantile(resolution_arr,0.84)-np.quantile(resolution_arr,0.16))/2.
    }
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v)
                                for k, v in metrics_mean.items())
    print("- Eval metrics : " + metrics_string)
    return metrics_mean, resolution_hists


if __name__ == '__main__':
    """
        Evaluate the model on the test set.
    """
    # Load the parameters
    args = parser.parse_args()

    # fetch dataloaders
    dataloaders = data_loader.fetch_dataloader(data_dir=osp.join(os.environ['PWD'],args.data), 
                                               batch_size=40, 
                                               validation_split=0.2)
    test_dl = dataloaders['test']

    # Define the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = net.Net(8, 3).to(device) #include puppi
    #model = net.Net(7, 3).to(device) #remove puppi
    optimizer = torch.optim.AdamW(model.parameters(),lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=500, threshold=0.05)

    loss_fn = net.loss_fn
    metrics = net.metrics
    model_dir = osp.join(os.environ['PWD'],args.ckpts)
    deltaR = 0.4
    deltaR_dz = 0.3

    # Reload weights from the saved file
    restore_ckpt = osp.join(model_dir, args.restore_file + '.pth.tar')
    ckpt = utils.load_checkpoint(restore_ckpt, model, optimizer, scheduler)
    epoch = ckpt['epoch']
    #utils.load_checkpoint(os.path.join(model_dir, args.restore_file + '.pth.tar'), model)
    with open(osp.join(model_dir, 'metrics_val_best.json')) as restore_metrics:
            best_validation_loss = json.load(restore_metrics)['loss']

    # Evaluate
    test_metrics, resolutions = evaluate(model, device, loss_fn, test_dl, metrics, deltaR, deltaR_dz, model_dir)
    validation_loss = test_metrics['loss']
    is_best = (validation_loss<best_validation_loss)
    if is_best: 
        print('Found new best loss!') 
        best_validation_loss=validation_loss
        # Save weights
        #utils.save_checkpoint({'epoch': epoch,
        #                       'state_dict': model.state_dict(),
        #                       'optim_dict': optimizer.state_dict(),
        #                       'sched_dict': scheduler.state_dict()},
        #                       is_best=True,
        #                       checkpoint=model_dir)
        # Save best val metrics in a json file in the model directory
        #utils.save_dict_to_json(test_metrics, osp.join(model_dir, 'metrics_val_best.json'))
        #utils.save(resolutions, osp.join(model_dir, 'best.resolutions'))

    utils.save(resolutions, os.path.join(model_dir, "{}.resolutions".format(args.restore_file)))
