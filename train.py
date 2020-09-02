import os.path as osp
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import MNISTSuperpixels
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from tqdm import tqdm
import utils
import model.net as net
import model.data_loader as data_loader
import warnings
warnings.simplefilter('ignore')


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
    dataloaders = data_loader.fetch_dataloader(data_dir=osp.join(os.environ['PWD'],'data'), 
                                               batch_size=60, 
                                               validation_split=.1)
    train_dl = dataloaders['train']
    
    model = net.Net().to('cuda')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-3)
    
    loss_fn = net.loss_fn
    model_dir = osp.join(os.environ['PWD'],'ckpts')

    for epoch in range(1,201):
        train(model, optimizer, loss_fn, train_dl, epoch)
        is_best=False
        utils.save_checkpoint({'epoch': epoch,
                               'state_dict': model.state_dict(),
                               'optim_dict': optimizer.state_dict()},
                              is_best=is_best,
                              checkpoint=model_dir)
