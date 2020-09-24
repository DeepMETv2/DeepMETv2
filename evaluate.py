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

parser = argparse.ArgumentParser()
parser.add_argument('--restore_file', default='best', help="name of the file in --model_dir \
                     containing weights to load")


def evaluate(model, loss_fn, dataloader, metrics):
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
    u_par_arr = []
    u_perp_arr = []
    qT_arr = []

    # compute metrics over the dataset
    for data in dataloader:

        data = data.to('cuda')

        # compute model output
        result = model(data)
        loss = loss_fn(result, data.y)

        # compute all metrics on this batch
        u_perp, u_par, qT = metrics['resolution'](result, data.y)
        u_perp_arr=np.concatenate((u_perp_arr,u_perp))
        u_par_arr=np.concatenate((u_par_arr,u_par))
        qT_arr=np.concatenate((qT_arr,qT))
        loss_avg_arr.append(loss.item())

    # compute mean of all metrics in summary
    bin_edges=np.arange(0,500,25)
    inds=np.digitize(qT_arr,bin_edges)
    qT_hist=[]
    u_perp_hist=[]
    u_par_hist=[]
    for i in range(1, len(bin_edges)):
        u_perp_i=u_perp_arr[np.where(inds==i)[0]]
        u_perp_hist.append((np.quantile(u_perp_i,0.84)-np.quantile(u_perp_i,0.16))/2.)
        u_par_i=u_par_arr[np.where(inds==i)[0]]
        u_par_hist.append((np.quantile(u_par_i,0.84)-np.quantile(u_par_i,0.16))/2.)
        qT_hist.append((bin_edges[i]+bin_edges[i-1])/2.)

    u_perp_resolution=np.histogram(qT_hist, bins=20, range=(0,500), weights=u_perp_hist)
    u_par_resolution=np.histogram(qT_hist, bins=20, range=(0,500), weights=u_par_hist)
    resolutions = {
        'u_perp_resolution': u_perp_resolution,
        'u_par_resolution': u_par_resolution
    }

    metrics_mean = {
        'loss': np.mean(loss_avg_arr),
        #'resolution': (np.quantile(resolution_arr,0.84)-np.quantile(resolution_arr,0.16))/2.
    }
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v)
                                for k, v in metrics_mean.items())
    print("- Eval metrics : " + metrics_string)
    return metrics_mean, resolutions


if __name__ == '__main__':
    """
        Evaluate the model on the test set.
    """
    # Load the parameters
    args = parser.parse_args()

    # fetch dataloaders
    dataloaders = data_loader.fetch_dataloader(data_dir=osp.join(os.environ['PWD'],'data'), 
                                               batch_size=60, 
                                               validation_split=.1)
    test_dl = dataloaders['test']

    # Define the model
    model = net.Net().to('cuda')

    loss_fn = net.loss_fn
    metrics = net.metrics
    model_dir = osp.join(os.environ['PWD'],'ckpts')

    # Reload weights from the saved file
    utils.load_checkpoint(os.path.join(model_dir, args.restore_file + '.pth.tar'), model)

    # Evaluate
    test_metrics, resolutions = evaluate(model, loss_fn, test_dl, metrics)
    save_path = os.path.join(model_dir, "metrics_val_{}.json".format(args.restore_file))
    utils.save_dict_to_json(test_metrics, save_path)
    utils.save(resolutions, os.path.join(model_dir, "{}.resolutions".format(args.restore_file)))
