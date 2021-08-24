import lz4.frame
import cloudpickle
import json
import os.path as osp
import os
import shutil

from numpy import pi

import torch
#from torch_cluster import radius

class RunningAverage():
    """A simple class that maintains the running average of a quantity
    
    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """
    def __init__(self):
        self.steps = 0
        self.total = 0
    
    def update(self, val):
        self.total += val
        self.steps += 1
    
    def __call__(self):
        return self.total/float(self.steps)

def load(filename):
    '''Load a coffea file from disk
    '''
    with lz4.frame.open(filename) as fin:
        output = cloudpickle.load(fin)
    return output


def save(output, filename):
    '''Save a coffea object or collection thereof to disk
    This function can accept any picklable object.  Suggested suffix: ``.coffea``
    '''
    with lz4.frame.open(filename, 'wb') as fout:
        thepickle = cloudpickle.dumps(output)
        fout.write(thepickle)

def save_dict_to_json(d, json_path):
    """Saves dict of floats in json file
    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)

def save_checkpoint(state, is_best, checkpoint):
    """Saves model and training parameters at checkpoint + 'last.pth.tar'. If is_best==True, also saves
    checkpoint + 'best.pth.tar'
    Args:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
        is_best: (bool) True if it is the best model seen till now
        checkpoint: (string) folder where parameters are to be saved
    """
    #ckpt = 'ckpt_{0}.pth.tar'.format(state['epoch'])
    #filepath = osp.join(checkpoint, ckpt)
    filepath = osp.join(checkpoint, 'last.pth.tar')
    if is_best:
        filepath = osp.join(checkpoint, 'best.pth.tar')
    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint))
        os.mkdir(checkpoint)
    else:
        print("Checkpoint Directory exists! ")
    torch.save(state, filepath)
    #if is_best:
    #    shutil.copyfile(filepath, osp.join(checkpoint, 'best.pth.tar'))


def load_checkpoint(checkpoint, model, optimizer=None, scheduler=None):
    """Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.
    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """
    if not os.path.exists(checkpoint):
        raise("File doesn't exist {}".format(checkpoint))
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint['state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])

    if scheduler:
        scheduler.load_state_dict(checkpoint['sched_dict'])

    return checkpoint
'''
def radius_graph_v2(etaphi, r , batch=None, loop=False, max_num_neighbors=32, flow='source_to_target', device='cuda'): # Problem: Max number is exceeded, this is incredibly slow
    etaphi_2pi = etaphi + torch.tensor([0,2*pi]).to(device)

    row, col = radius(etaphi, etaphi, r, batch, batch, max_num_neighbors + 1)
    row1, col1 = radius(etaphi, etaphi_2pi, r, batch, batch, max_num_neighbors +1)

    #row = torch.cat((row, row1), 0)
    #col = torch.cat((col, col1), 0)

    assert flow in ['source_to_target', 'target_to_source']
    row, col = (col, row) if flow == 'source_to_target' else (row, col)
    if not loop:
        mask = row != col
        row, col = row[mask], col[mask]
    edge_index = torch.stack([row,col], dim=0)
    #edge_index = torch.unique(edge_index, dim=0)
    
    return edge_index
'''