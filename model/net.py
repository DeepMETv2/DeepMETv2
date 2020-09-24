"""Defines the neural network, loss function and metrics"""

import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.dynamic_reduction_network import DynamicReductionNetwork

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.drn = DynamicReductionNetwork(input_dim=10, hidden_dim=64,
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
                                                              1./7.              #fromPV
                                                          ]))
    def forward(self, data):
        output = self.drn(data)
        met    = nn.Softplus()(output[:,0]).unsqueeze(1)
        metphi = math.pi*(2*torch.sigmoid(output[:,1]) - 1).unsqueeze(1)
        #metphi = nn.Hardtanh(-math.pi, math.pi)(output[:,1]).unsqueeze(1)
        #output = torch.cat((x*torch.cos(y), x*torch.sin(y)), 1)
        output = torch.cat((met, metphi), 1)  
        return output

def loss_fn(prediction, truth):

    qTx= truth[:,0]*torch.cos(truth[:,1])
    qTy= truth[:,0]*torch.sin(truth[:,1])
    qT = truth[:,0]
    METx = prediction[:,0]*torch.cos(prediction[:,1])
    METy = prediction[:,0]*torch.sin(prediction[:,1])
    MET  = prediction[:,0]
    
    loss = ( F.mse_loss(MET, qT, reduction='mean') +
             F.mse_loss(METx, qTx, reduction='mean') +
             F.mse_loss(METy, qTy, reduction='mean') ) / 3.
    return loss.mean()

def resolution(prediction, truth):
    
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

    METx = prediction[:,0]*torch.cos(prediction[:,1])
    METy = prediction[:,0]*torch.sin(prediction[:,1])
    # predicted MET/qT
    v_MET=torch.stack((METx, METy),dim=1)

    response = getdot(v_MET,v_qT)/getdot(v_qT,v_qT)
    v_paral_predict = scalermul(response, v_qT)
    u_paral_predict = getscale(v_paral_predict)-getscale(v_qT)
    u_paral_predict = u_paral_predict
    v_perp_predict = v_MET - v_paral_predict
    u_perp_predict = getscale(v_perp_predict)
    u_perp_predict = u_perp_predict
    return u_perp_predict.cpu().detach().numpy(), u_paral_predict.cpu().detach().numpy(), truth[:,0].cpu().detach().numpy(), response.cpu().detach().numpy()

# maintain all metrics required in this dictionary- these are used in the training and evaluation loops
metrics = {
    'resolution': resolution,
}
