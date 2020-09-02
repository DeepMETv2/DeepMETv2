"""
    PyTorch specification for the hit graph dataset.
"""

# System imports
import os
import glob
import os.path as osp

# External imports
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import random_split
from torch_geometric.utils import is_undirected, to_undirected
from torch_geometric.data import (Data, Dataset)
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

class METDataset(Dataset):
    """PyTorch geometric dataset from processed hit information"""
    
    def __init__(self, root):
        super(METDataset, self).__init__(root)
    
    def download(self):
        pass #download from xrootd or something later
    
    @property
    def raw_file_names(self):
        if not hasattr(self,'input_files'):
            self.input_files = glob.glob(self.raw_dir+'/*.npz')
        return [f.split('/')[-1] for f in self.input_files]
    
    @property
    def processed_file_names(self):
        if not hasattr(self,'processed_files'):
            proc_names = ['data_{}.pt'.format(idx) for idx in range(len(self.raw_file_names))]
            self.processed_files = [osp.join(self.processed_dir,name) for name in proc_names]
        return self.processed_files
    
    def __len__(self):
        return len(self.processed_file_names)
    
    def get(self, idx):
        data = torch.load(self.processed_files[idx])
        return data
    
    def process(self):
        #convert the npz into pytorch tensors and save them
        path = self.processed_dir
        for idx,raw_path in enumerate(tqdm(self.raw_paths)):
            npzfile = np.load(raw_path)
            x = npzfile['arr_0'].astype(np.float32)
            edge_index = torch.empty((2,0), dtype=torch.long)
            metx=npzfile['arr_1'][0]*np.cos(npzfile['arr_1'][1])
            mety=npzfile['arr_1'][0]*np.sin(npzfile['arr_1'][1])
            y=np.array([metx,mety]).astype(np.float32)[None]
            outdata = Data(x=torch.from_numpy(x),
                           edge_index=edge_index,
                           y=torch.from_numpy(y))
        
            torch.save(outdata, osp.join(self.processed_dir, 'data_{}.pt'.format(idx)))

def fetch_dataloader(data_dir, batch_size, validation_split):
    transform = T.Cartesian(cat=False)
    dataset = METDataset(data_dir)
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    random_seed= 42
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    dataloaders = {
        'train':  DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(train_indices)),
        'test':   DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(val_indices))
        }
    return dataloaders
