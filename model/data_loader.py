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
            self.input_files = sorted(glob.glob(self.raw_dir+'/*.npz'))
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
            inputs = npzfile['arr_0'].astype(np.float32)
            pX=inputs[:,0]*np.cos(inputs[:,2])
            pY=inputs[:,0]*np.sin(inputs[:,2])
            pT=inputs[:,0]
            eta=inputs[:,1]
            d0=inputs[:,4]
            dz=inputs[:,5]
            #dz[dz==float('inf')]=589 #there is one particle with dz=inf
            mass=inputs[:,3]
            pdgId=inputs[:,6]
            charge=inputs[:,7]
            fromPV=inputs[:,8]
            puppiWeight=inputs[:,9]
            x = np.stack((pX,pY,pT,eta,d0,dz,mass,puppiWeight,pdgId,charge,fromPV),axis=-1)
            x = np.nan_to_num(x)
            x = np.clip(x, -5000., 5000.)
            #x = npzfile['arr_0'][:,:4].astype(np.float32)
            edge_index = torch.empty((2,0), dtype=torch.long)
            y = npzfile['arr_1'].astype(np.float32)[None]
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
    print(split)
    random_seed= 42
    train_subset, val_subset = torch.utils.data.random_split(dataset, [dataset_size - split, split],
                                                             generator=torch.Generator().manual_seed(random_seed))
    print(len(train_subset), len(val_subset))
    dataloaders = {
        'train':  DataLoader(train_subset, batch_size=batch_size, shuffle=False),
        'test':   DataLoader(val_subset, batch_size=batch_size, shuffle=False)
        }
    return dataloaders
