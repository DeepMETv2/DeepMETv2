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
import time

import matplotlib.pyplot as plt

warnings.simplefilter('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before \
                    training")  # 'best' or 'last'
"""
parser.add_argument('--data', default='data',
                    help="Name of the data folder")
parser.add_argument('--ckpts', default='ckpts',
                    help="Name of the ckpts folder")
"""
parser.add_argument('--config', default = 'config.yaml',
                    help="Name of config")


def train(model, device, optimizer, scheduler, loss_fn, dscb_loss_fn, dataloader, epoch):
    model.train()
    loss_avg_arr = []
    loss_avg = utils.RunningAverage()
    start_time = time.time()

    with tqdm(total=len(dataloader)) as t:
        for data in dataloader:
            optimizer.zero_grad()
            data = data.to(device)

            #dz = data.x[:,5] 
            # NB: there is a problem right now for comparing hits at the +/- pi boundary
            #edge_index_phi = utils.radius_graph_v2(etaphi, r=deltaR, batch=data.batch, loop=True, max_num_neighbors=10, device=device)
            #print(edge_index_simple)
            #edge_index_dz = radius_graph(dz, r=deltaR_dz, batch=data.batch, loop=True, max_num_neighbors=255)
            #tinf = (torch.ones(len(dz))*float("Inf")).to('cuda')
            #edge_index_dz = knn_graph(torch.where(data.x[:,7]!=0, dz, tinf), k=deltaR_dz, batch=data.batch, loop=True)
            #cat_edges = torch.cat([edge_index,edge_index_dz],dim=1)
            fixed = train_info["mode"]["fixed"]

            if fixed:
                if train_info["max_num_neighbors"] == 0:
                    edge_index_simple = torch.arange(len(data.x[:,1]))
                    edge_index = edge_index_simple.expand(2, len(data.x[:,1])).to(device)
                else:
                    phi = torch.atan2(data.x[:,1], data.x[:,0])
                    etaphi = torch.cat([data.x[:,3][:,None], phi[:,None]], dim=1) 
                    edge_index = radius_graph(etaphi, r=deltaR, batch=data.batch, loop=train_info["loop"], max_num_neighbors=train_info["max_num_neighbors"])

            if not fixed:
                edge_index = None

            # result = model(x_cont, x_cat, None, data.batch)
            # result = model(data.x, edge_index=edge_index, batch=data.batch)
            result, dscb_params = model(data.x, edge_index=edge_index, batch=data.batch)
            #print(dscb_params.cpu().detach().numpy())
            #print(dscb_params)
            #print(result.shape)
            #print(data.x.size())
            # --
            #loss = loss_fn(result, data.x, data.y, data.batch)
            #print(loss)
            #loss.backward()
            
            loss = dscb_loss_fn(result, data.x, data.y, data.batch, dscb_params)
            #print("loss: ",loss)
            loss.sum().backward()

            optimizer.step()
            # update the average loss
            loss_avg_arr.append(loss.sum().cpu().item())
            loss_avg.update(loss.sum().cpu().item())
            t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
            t.update()
            #torch.cuda.empty_cache()

    training_time = time.time() - start_time
    mean_loss = np.mean(loss_avg_arr)
    scheduler.step(mean_loss)
    print('Training epoch: {:02d}, MSE: {:.4f}'.format(epoch, mean_loss))
    return mean_loss, training_time
    

if __name__ == '__main__':
    args = parser.parse_args()

    info_handler = utils.info_handler(args.config)
    train_info = info_handler.get_info("train")
    net_info = info_handler.get_info("net")
    graph_info = info_handler.get_info("graph_met_network")

    print(train_info["mode"])


    dataloaders = data_loader.fetch_dataloader(data_dir=osp.join(os.environ['PWD'],train_info["dataset"]), 
                                               batch_size=train_info["batch_size"],
                                               validation_split=train_info["validation_split"])
    train_dl = dataloaders['train']
    test_dl = dataloaders['test']

    print(len(train_dl), len(test_dl))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    print(device)   
    torch.cuda.set_per_process_memory_fraction(0.5)
    #model = net.Net(8, 3).to('cuda')
    model = net.Net(net_info["continuous_dim"], net_info["categorical_dim"], net_info["output_dim"], net_info["hidden_dim"], net_info["conv_depth"], train_info["mode"], graph_info["k"], nn.ELU()).to(device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=train_info["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=train_info["factor"], patience=train_info["patience"], threshold=train_info["threshold"])
    first_epoch = 0
    best_validation_loss = 10e7
    deltaR = train_info["deltaR"]
    deltaR_dz = 10

    loss_fn = net.loss_fn
    dscb_loss_fn = net.dscb_loss_fn
    metrics = net.metrics

    model_dir = osp.join(os.environ['PWD'],train_info["ckpt"])

    # reload weights from restore_file if specified
    if args.restore_file is not None:
        restore_ckpt = osp.join(model_dir, args.restore_file + '.pth.tar')
        #print(restore_ckpt)
        ckpt = utils.load_checkpoint(restore_ckpt, model, optimizer, scheduler)
        first_epoch = ckpt['epoch']
        print('Restarting training from epoch',first_epoch)
        with open(osp.join(model_dir, 'metrics_val_best.json')) as restore_metrics:
            best_validation_loss = json.load(restore_metrics)['loss']
    

    for epoch in range(first_epoch+1,31):

        print('Current best loss:', best_validation_loss)
        if '_last_lr' in scheduler.state_dict():
            print('Learning rate:', scheduler.state_dict()['_last_lr'][0])

        # compute number of batches in one epoch (one full pass over the training set)
        training_loss, training_time = train(model, device, optimizer, scheduler, loss_fn, dscb_loss_fn, train_dl, epoch)

        # Save weights
        utils.save_checkpoint({'epoch': epoch,
                               'state_dict': model.state_dict(),
                               'optim_dict': optimizer.state_dict(),
                               'sched_dict': scheduler.state_dict()},
                              is_best=False,
                              checkpoint=model_dir)

        # Evaluate for one epoch on validation set
        test_metrics, resolutions, evaluation_time = evaluate(model, device, loss_fn, dscb_loss_fn, test_dl, metrics, deltaR, deltaR_dz, model_dir, out='', train_info = train_info)

        validation_loss = test_metrics['loss']

        is_best = (validation_loss<=best_validation_loss)

        # If best_eval, best_save_path
        if is_best: 
            print('Found new best loss!') 
            best_validation_loss=validation_loss

            # Save weights
            utils.save_checkpoint({'epoch': epoch,
                                   'state_dict': model.state_dict(),
                                   'optim_dict': optimizer.state_dict(),
                                   'sched_dict': scheduler.state_dict()},
                                  is_best=True,
                                  checkpoint=model_dir)
            
            # Save best val metrics in a json file in the model directory
            utils.save_dict_to_json(test_metrics, osp.join(model_dir, 'metrics_val_best.json'))
            utils.save(resolutions, osp.join(model_dir, 'best.resolutions'))

        utils.save_dict_to_json(test_metrics, osp.join(model_dir, 'metrics_val_last.json'))
        utils.save(resolutions, osp.join(model_dir, 'last.resolutions'))


        info_handler.add_info("sample", str(train_info["dataset"]))
        info_handler.add_epoch(epoch, float(training_loss), float(validation_loss), float(training_time), float(evaluation_time))
        info_handler.save_infos(str(model_dir))
        info_handler.plot_loss(str(model_dir))

