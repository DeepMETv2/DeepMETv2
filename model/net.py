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
        self.drn = DynamicReductionNetwork(input_dim=4, hidden_dim=64,
                                           k = 8,
                                           output_dim=2, aggr='max',
                                           norm=torch.tensor([1./2950.0, 1./6., 1./3.1416016, 1./1.2050781]))
    def forward(self, data):
        output = self.drn(data)
        softplus=nn.Softplus()
        x=softplus(output[:,0]).unsqueeze(1)
        hardtanh=nn.Hardtanh(-math.pi, math.pi)
        y=hardtanh(output[:,1]).unsqueeze(1)
        output = torch.cat((x, torch.cos(y), torch.sin(y)), 1)
        #output = torch.cat((x, y), 1)  
        return output

def loss_fn(prediction, truth):

    METx = prediction[:,0]*prediction[:,1]
    METy = prediction[:,0]*prediction[:,2]
    MET  = prediction[:,0]#torch.sqrt( METx**2 + METy**2 )
    loss = (MET - truth[:,0])**2
    loss += (METx - (truth[:,0]*torch.cos(truth[:,1])))**2
    loss += (METy - (truth[:,0]*torch.sin(truth[:,1])))**2
    loss /= 3.
    return loss.mean()
    #return F.mse_loss(prediction[:,0], truth[:,0], reduction='mean')

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

    METx = prediction[:,0]*prediction[:,1]
    METy = prediction[:,0]*prediction[:,2]
    # predicted MET/qT
    v_MET=torch.stack((METx, METy),dim=1)

    response = getdot(v_MET,v_qT)/getdot(v_qT,v_qT)
    v_paral_predict = scalermul(response, v_qT)
    v_perp_predict = v_MET - v_paral_predict
    u_perp_predict = getscale(v_perp_predict)
    return u_perp_predict.cpu().detach().numpy()

# maintain all metrics required in this dictionary- these are used in the training and evaluation loops
metrics = {
    'resolution': resolution,
}
