"""Defines the neural network, loss function and metrics"""

import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add
from model.dynamic_reduction_network import DynamicReductionNetwork
from model.graph_met_network import GraphMETNetwork

'''
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.drn = DynamicReductionNetwork(input_dim=11, hidden_dim=64,
                                           k = 8,
                                           output_dim=2, aggr='max',
                                           norm=torch.tensor([1./2950.0,         #px
                                                              1./2950.0,         #py
                                                              1./2950.0,         #pt
                                                              1./5.265625,       #eta
                                                              1./143.875,        #d0
                                                              1./589.,           #dz
                                                              1./1.2050781,      #mass
                                                              1./211.,           #pdgId
                                                              1.,                #charge
                                                              1./7.,             #fromPV
                                                              1.                 #puppiWeight
                                                          ]))
    def forward(self, data):
        output = self.drn(data)
        met    = nn.Softplus()(output[:,0]).unsqueeze(1)
        metphi = math.pi*(2*torch.sigmoid(output[:,1]) - 1).unsqueeze(1)
        output = torch.cat((met, metphi), 1)  
        return output
'''
class Net(nn.Module):
    def __init__(self, continuous_dim, categorical_dim):
        super(Net, self).__init__()
        self.graphnet = GraphMETNetwork(continuous_dim, categorical_dim,
                                        output_dim=1, hidden_dim=64,
                                        conv_depth=2)
    
    def forward(self, x_cont, x_cat, edge_index, batch):
        weights = self.graphnet(x_cont, x_cat, edge_index, batch)
        return torch.sigmoid(weights)

def loss_fn(weights, prediction, truth, batch):
    '''
    qTx= truth[:,0]*torch.cos(truth[:,1])
    qTy= truth[:,0]*torch.sin(truth[:,1])
    qT = truth[:,0]
    qTphi= truth[:,1]
    METx = prediction[:,0]*torch.cos(prediction[:,1])
    METy = prediction[:,0]*torch.sin(prediction[:,1])
    MET  = prediction[:,0]
    METphi = prediction[:,1]
    lossMET    = (MET-qT)**2/qT
    lossMETphi = 1.-torch.cos(METphi-qTphi)
    
    #loss = ( F.mse_loss(MET, qT, reduction='mean') +
    #loss = ( F.mse_loss(MET, qT, reduction='mean') +
    #F.mse_loss(METy, qTy, reduction='mean') ) / 2.
    loss = torch.sqrt(lossMET.mean()*lossMETphi.mean())
    '''
    true_MET = truth[:,0]
    true_METphi = truth[:,1]
    px=prediction[:,0]
    py=prediction[:,1]
    true_px=true_MET*torch.cos(true_METphi)
    true_py=true_MET*torch.sin(true_METphi)
    METx = scatter_add(weights*px, batch)
    METy = scatter_add(weights*py, batch)
    loss=0.5*( ( METx + true_px)**2 + ( METy + true_py)**2 ).mean()
    return loss

def resolution(weights, prediction, truth, batch):
    
    def getdot(vx, vy):
        return torch.einsum('bi,bi->b',vx,vy)
    def getscale(vx):
        return torch.sqrt(getdot(vx,vx))
    def scalermul(a,v):
        return torch.einsum('b,bi->bi',a,v)    

    qTx=truth[:,0]*torch.cos(truth[:,1])
    qTy=truth[:,0]*torch.sin(truth[:,1])
    # truth qT
    v_qT=torch.stack((qTx,qTy),dim=1)

    pfMETx=truth[:,2]*torch.cos(truth[:,3])
    pfMETy=truth[:,2]*torch.sin(truth[:,3])
    # PF MET
    v_pfMET=torch.stack((pfMETx, pfMETy),dim=1)

    puppiMETx=truth[:,4]*torch.cos(truth[:,5])
    puppiMETy=truth[:,4]*torch.sin(truth[:,5])
    # PF MET                                                                                                                                                            
    v_puppiMET=torch.stack((puppiMETx, puppiMETy),dim=1)

    px=prediction[:,0]
    py=prediction[:,1]
    METx = scatter_add(weights*px, batch)
    METy = scatter_add(weights*py, batch)
    # predicted MET/qT
    v_MET=torch.stack((METx, METy),dim=1)

    def compute(vector):
        response = getdot(vector,v_qT)/getdot(v_qT,v_qT)
        v_paral_predict = scalermul(response, v_qT)
        u_paral_predict = getscale(v_paral_predict)-getscale(v_qT)
        v_perp_predict = vector - v_paral_predict
        u_perp_predict = getscale(v_perp_predict)
        return [u_perp_predict.cpu().detach().numpy(), u_paral_predict.cpu().detach().numpy(), response.cpu().detach().numpy()]

    resolutions= {
        'MET':      compute(v_MET),
        'pfMET':    compute(v_pfMET),
        'puppiMET': compute(v_puppiMET)
    }
    return resolutions, truth[:,0].cpu().detach().numpy()

# maintain all metrics required in this dictionary- these are used in the training and evaluation loops
metrics = {
    'resolution': resolution,
}
