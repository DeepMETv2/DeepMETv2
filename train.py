import json
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


def train(model, optimizer, scheduler, loss_fn, dataloader, epoch):
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
    scheduler.step(np.mean(loss_avg_arr))
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
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2, threshold=0.05)
    first_epoch = 0
    best_validation_loss = 10e7

    loss_fn = net.loss_fn
    metrics = net.metrics

    model_dir = osp.join(os.environ['PWD'],'ckpts')

    # reload weights from restore_file if specified
    if args.restore_file is not None:
        restore_ckpt = osp.join(model_dir, args.restore_file + '.pth.tar')
        ckpt = utils.load_checkpoint(restore_ckpt, model, optimizer, scheduler)
        first_epoch = ckpt['epoch']
        print('Restarting training from epoch',first_epoch)
        with open(osp.join(model_dir, 'metrics_val_best.json')) as restore_metrics:
            best_validation_loss = json.load(restore_metrics)['loss']

    for epoch in range(first_epoch+1,201):

        print('Current best loss:', best_validation_loss)
        if '_last_lr' in scheduler.state_dict():
            print('Learning rate:', scheduler.state_dict()['_last_lr'][0])

        # compute number of batches in one epoch (one full pass over the training set)
        train(model, optimizer, scheduler, loss_fn, train_dl, epoch)

        # Evaluate for one epoch on validation set
        test_metrics, resolutions = evaluate(model, loss_fn, test_dl, metrics)

        validation_loss = test_metrics['loss']
        is_best = (validation_loss<=best_validation_loss)

        # Save weights
        utils.save_checkpoint({'epoch': epoch,
                               'state_dict': model.state_dict(),
                               'optim_dict': optimizer.state_dict(),
                               'sched_dict': scheduler.state_dict()},
                              is_best=is_best,
                              checkpoint=model_dir)

        # If best_eval, best_save_path
        if is_best: 
            print('Found new best loss!') 
            best_validation_loss=validation_loss

            # Save best val metrics in a json file in the model directory
            utils.save_dict_to_json(test_metrics, osp.join(model_dir, 'metrics_val_best.json'))
            utils.save(resolutions, osp.join(model_dir, 'best.resolutions'))

        utils.save_dict_to_json(test_metrics, osp.join(model_dir, 'metrics_val_last.json'))
        utils.save(resolutions, osp.join(model_dir, 'last.resolutions'))
