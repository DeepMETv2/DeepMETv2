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
            #self.input_files = sorted(glob.glob(self.raw_dir+'/tt_file0_*.npz'))
            self.input_files = sorted(glob.glob(self.raw_dir+'/*.npz'))
        return [f.split('/')[-1] for f in self.input_files]

    @property
    def existing_pt_names(self):
        if not hasattr(self,'pt_files'):
            self.pt_files = sorted(glob.glob(self.processed_dir+'/*file*slice*nevent*pt'))
        return [f.split('/')[-1] for f in self.pt_files]
    
    @property
    def processed_file_names(self):
        if not hasattr(self,'processed_files'):
            #print(self.raw_file_names)
            #FIXME need to figure out how to get a list of expected pt files
            proc_names = [idx for idx in self.existing_pt_names]
            #proc_names = ['data_{}.pt'.format(idx) for idx in range(len(self.raw_file_names))]
            self.processed_files = [osp.join(self.processed_dir,name) for name in proc_names]
        return self.processed_files
    
    def __len__(self):
        return len(self.processed_file_names)
   
    def len(self):
        return len(self.processed_file_names)
 
    def get(self, idx):
        data = torch.load(self.processed_files[idx])
        return data
    
    def process(self):
        #convert the npz into pytorch tensors and save them
        path = self.processed_dir
        for idx,raw_path in enumerate(tqdm(self.raw_paths)):
            npzfile = np.load(raw_path,allow_pickle=True)
            for ievt in range(np.shape(npzfile['x'])[1]):
                inputs = np.array(npzfile['x'][:,ievt,:]).astype(np.float32)
                #original: pt, eta, phi, d0, dz, mass, puppiWeight, pdgid, charge, frompv, pvref, pvAssocQuality
                inputs=inputs.T 
                #now: pX,pY,pT,eta,d0,dz,mass,puppiWeight,pdgId,charge,fromPV
                x = inputs[:,3:10]
                x=np.insert(x,0, inputs[:,0]*np.cos(inputs[:,2]),axis=1)
                x=np.insert(x,1, inputs[:,0]*np.sin(inputs[:,2]),axis=1)
                x=np.insert(x,2, inputs[:,0],axis=1)
                x=np.insert(x,3, inputs[:,1],axis=1)
                x=x[x[:,8]!=-999]
                x=x[x[:,9]!=-999]
                #print(x[0])
                x = np.nan_to_num(x)
                x = np.clip(x, -5000., 5000.)
                assert not np.any(np.isnan(x))
                edge_index = torch.empty((2,0), dtype=torch.long)
                y = (np.array(npzfile['y'][ievt,:]).astype(np.float32)[None])
                #print(y)
                outdata = Data(x=torch.from_numpy(x),
                               edge_index=edge_index,
                               y=torch.from_numpy(y))
                torch.save(outdata, osp.join(self.processed_dir,(raw_path.replace('.npz','_'+str(ievt)+'.pt')).split('/')[-1] ))

def fetch_dataloader(data_dir, batch_size, validation_split):
    transform = T.Cartesian(cat=False)
    dataset = METDataset(data_dir)
    #print(dataset)
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    print(split)
    random_seed = 42
    # fix the random generator for train and test
    # taken from https://pytorch.org/docs/1.5.0/notes/randomness.html
    torch.manual_seed(random_seed)
    train_subset, val_subset = torch.utils.data.random_split(dataset, [dataset_size - split, split])#,
#                                                             generator=torch.Generator().manual_seed(random_seed))
    print('length of train/val data: ', len(train_subset), len(val_subset))
    dataloaders = {
        'train':  DataLoader(train_subset, batch_size=batch_size, shuffle=False),
        'test':   DataLoader(val_subset, batch_size=batch_size, shuffle=False)
        }
    return dataloaders





