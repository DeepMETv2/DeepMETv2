import os.path as osp
import os
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
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

parser = argparse.ArgumentParser()
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before \
                    training")  # 'best' or 'train'


def train(model, optimizer, loss_fn, dataloader, epoch):
    model.train()
    loss_avg_arr = []
    loss_avg = utils.RunningAverage()
    with tqdm(total=len(dataloader)) as t:
        for data in dataloader:
            data = data.to('cuda')
            optimizer.zero_grad()
            result = model(data)
            loss = loss_fn(result, data.y)
            loss.backward()
            optimizer.step()
            # update the average loss
            loss_avg_arr.append(loss.item())
            loss_avg.update(loss.item())
            t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
            t.update()
    print('Training epoch: {:02d}, MSE: {:.4f}'.format(epoch, np.mean(loss_avg_arr)))

if __name__ == '__main__':
    args = parser.parse_args()

    dataloaders = data_loader.fetch_dataloader(data_dir=osp.join(os.environ['PWD'],'data'), 
                                               batch_size=60, 
                                               validation_split=.1)
    train_dl = dataloaders['train']
    test_dl = dataloaders['test']

    model = net.Net().to('cuda')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-3)
    loss_fn = net.loss_fn
    metrics = net.metrics

    model_dir = osp.join(os.environ['PWD'],'ckpts')
    # reload weights from restore_file if specified
    if args.restore_file is not None:
        restore_path = osp.join(model_dir, args.restore_file + '.pth.tar')
        utils.load_checkpoint(restore_path, model, optimizer)

    best_val_res = 10e7
    for epoch in range(1,201):
        train(model, optimizer, loss_fn, train_dl, epoch)
        test_metrics = evaluate(model, loss_fn, test_dl, metrics)
        val_res = test_metrics['resolution']
        is_best = (val_res<=best_val_res)
        utils.save_checkpoint({'epoch': epoch,
                               'state_dict': model.state_dict(),
                               'optim_dict': optimizer.state_dict()},
                              is_best=is_best,
                              checkpoint=model_dir)
