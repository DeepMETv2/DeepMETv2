"""Evaluates the model"""

import argparse
import logging
import os.path as osp
import os

import numpy as np
import torch
from torch.autograd import Variable

import utils
import model.net as net
import model.data_loader as data_loader

from torch_geometric.utils import to_undirected
from torch_cluster import radius_graph, knn_graph

import matplotlib.pyplot as plt
import mplhep as hep
plt.style.use(hep.style.CMS)


parser = argparse.ArgumentParser()
parser.add_argument('--restore_file', default='best', help="name of the file in --model_dir \
                     containing weights to load")
parser.add_argument('--data', default='data',
                    help="Name of the data folder")
parser.add_argument('--ckpts', default='ckpts',
                    help="Name of the ckpts folder")
parser.add_argument('--save_plot', default=True,
                    help="save_plot while training")

def evaluate(model, loss_fn, dataloader, metrics, deltaR, model_dir, save_plot):
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
        'MET': 'Graph MET'
    }

    # compute metrics over the dataset
    for data in dataloader:

        has_deepmet = (data.y.size()[1] > 6)
        
        if has_deepmet == True and 'deepMETResponse' not in resolutions_arr.keys():
            resolutions_arr.update({
                'deepMETResponse': [[],[],[]],
                'deepMETResolution': [[],[],[]]
            })
        
        data = data.to('cuda')
        x_cont = data.x[:,:8]
        x_cat = data.x[:,8:].long()
        phi = torch.atan2(data.x[:,1], data.x[:,0])
        etaphi = torch.cat([data.x[:,3][:,None], phi[:,None]], dim=1)
        # NB: there is a problem right now for comparing hits at the +/- pi boundary                                                
        edge_index = radius_graph(etaphi, r=deltaR, batch=data.batch, loop=True, max_num_neighbors=255)
        # compute model output
        result = model(x_cont, x_cat, edge_index, data.batch)
        loss = loss_fn(result, data.x, data.y, data.batch)

        # compute all metrics on this batch
        resolutions, qT= metrics['resolution'](result, data.x, data.y, data.batch)
        for key in resolutions_arr:
            for i in range(len(resolutions_arr[key])):
                resolutions_arr[key][i]=np.concatenate((resolutions_arr[key][i],resolutions[key][i]))
        qT_arr=np.concatenate((qT_arr,qT))
        loss_avg_arr.append(loss.item())

    # compute mean of all metrics in summary
    bin_edges=np.arange(0,400,20)
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

        u_perp_resolution=np.histogram(qT_hist, bins=20, range=(0,400), weights=u_perp_hist)
        u_perp_scaled_resolution=np.histogram(qT_hist, bins=20, range=(0,400), weights=u_perp_scaled_hist)
        u_par_resolution=np.histogram(qT_hist, bins=20, range=(0,400), weights=u_par_hist)
        u_par_scaled_resolution=np.histogram(qT_hist, bins=20, range=(0,400), weights=u_par_scaled_hist)
        R=np.histogram(qT_hist, bins=20, range=(0,400), weights=R_hist)

        if(save_plot):
            plt.figure()
            plt.figure(1)
            plt.plot(qT_hist, u_perp_hist,        color=colors[key], label=labels[key])
            plt.figure(2)
            plt.plot(qT_hist, u_perp_scaled_hist, color=colors[key], label=labels[key])
            plt.figure(3)
            plt.plot(qT_hist, u_par_hist,         color=colors[key], label=labels[key])
            plt.figure(4)
            plt.plot(qT_hist, u_par_scaled_hist,  color=colors[key], label=labels[key])
            plt.figure(5)
            plt.plot(qT_hist, R_hist,             color=colors[key], label=labels[key])
            
        resolution_hists[key] = {
            'u_perp_resolution': u_perp_resolution,
            'u_perp_scaled_resolution': u_perp_scaled_resolution,
            'u_par_resolution': u_par_resolution,
            'u_par_scaled_resolution':u_par_scaled_resolution,
            'R': R
        }

    if(save_plot):
        plt.figure(1)
        plt.axis([0, 400, 0, 25])
        plt.xlabel(r'$q_{T}$ [GeV]')
        plt.ylabel(r'$\sigma (u_{\perp})$ [GeV]')
        plt.legend()
        plt.savefig(model_dir+'/resol_perp.png')
        plt.clf()
        plt.close()

        plt.figure(2)
        plt.axis([0, 400, 0, 30])
        plt.xlabel(r'$q_{T}$ [GeV]')
        plt.ylabel(r'Scaled $\sigma (u_{\perp})$ [GeV]')
        plt.legend()
        plt.savefig(model_dir+'/resol_perp_scaled.png')
        plt.clf()
        plt.close()

        plt.figure(3)
        plt.axis([0, 400, 0, 45])
        plt.xlabel(r'$q_{T}$ [GeV]')
        plt.ylabel(r'$\sigma (u_{\parallel})$ [GeV]')
        plt.legend()
        plt.savefig(model_dir+'/resol_parallel.png')
        plt.clf()
        plt.close()

        plt.figure(4)
        plt.axis([0, 400, 0, 60])
        plt.xlabel(r'$q_{T}$ [GeV]')
        plt.ylabel(r'Scaled $\sigma (u_{\parallel})$ [GeV]')
        plt.legend()
        plt.savefig(model_dir+'/resol_parallel_scaled.png')
        plt.clf()
        plt.close()

        plt.figure(5)
        plt.axis([0, 400, 0, 1.2])
        plt.axhline(y=1.0, color='black', linestyle='-.')
        plt.xlabel(r'$q_{T}$ [GeV]')
        plt.ylabel(r'Response $-\frac{<u_{\parallel}>}{<q_{T}>}$')
        plt.legend()
        plt.savefig(model_dir+'/response_parallel.png')
        plt.clf()
        plt.close()

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
                                               batch_size=60, 
                                               validation_split=.5)
    test_dl = dataloaders['test']

    # Define the model
    model = net.Net(8, 3).to('cuda')

    loss_fn = net.loss_fn
    metrics = net.metrics
    model_dir = osp.join(os.environ['PWD'],args.ckpts)
    deltaR = 0.4

    # Reload weights from the saved file
    utils.load_checkpoint(os.path.join(model_dir, args.restore_file + '.pth.tar'), model)

    # Evaluate
    test_metrics, resolutions = evaluate(model, loss_fn, test_dl, metrics, deltaR, model_dir, args.save_plot)
    #save_path = os.path.join(model_dir, "metrics_val_{}.json".format(args.restore_file))
    #utils.save_dict_to_json(test_metrics, save_path)
    utils.save(resolutions, os.path.join(model_dir, "{}.resolutions".format(args.restore_file)))
