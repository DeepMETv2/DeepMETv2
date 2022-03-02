import torch
import numpy as np
from torch.nn.functional import softplus

'''
constants
'''

sqrtPiOn2 = 1.25331413732
sqrt2 = 1.41421356237

minval = 1e-9
maxval = 1e9

'''
double-sided crystal ball function

three implementations, which just exist because that's the order I wrote them in.
The relevant one is "smarter" which is a smarter vectorized implementation than "naiive_vectorized" 
'''

def dscb_single(x, mu, sigma, alphaL, nL, alphaR, nR):
    t = (x-mu)/sigma
    
    fact1L = alphaL/nL
    fact2L = nL/alphaL - alphaL - t

    fact1R = alphaR/nR
    fact2R = nR/alphaR - alphaR + t

    if -alphaL <= t and alphaR >= t:
        result = torch.exp(-0.5*t*t)
    elif t < -alphaL:
        result = torch.exp(-0.5*alphaL*alphaL) * torch.pow(fact1L * fact2L, -nL)
    elif t > alphaR:
        result = torch.exp(-0.5*alphaR*alphaR) * torch.pow(fact1R * fact2R, -nR)
    else:
        print("UH OH")
        result = torch.zeros_like(x)

    return result

def naiive_vectorized(x, mu, sigma, alphaL, nL, alphaR, nR):
    result = torch.zeros_like(x)
    for i in range(len(x)):
        result[i] = dscb_single(x[i], mu[i], sigma[i], alphaL[i], nL[i], alphaR[i], nR[i])

    norm = double_crystalball_norm(mu, sigma, alphaL, nL, alphaR, nR)

    return result/norm

def smarter(x, mu, sigma, alphaL, nL, alphaR, nR):
    t = (x-mu)/sigma

    result = torch.empty_like(x)

    middle = torch.logical_and(-alphaL <= t, t <= alphaR)
    left = t < - alphaL
    right = alphaR < t

    tM = t[middle]
    result[middle] = torch.exp(-0.5*tM*tM)

    nLL = nL[left]
    tL = t[left]
    alphaLL = alphaL[left]
    fact1L = alphaLL/nLL
    fact2L = nLL/alphaLL - alphaLL - tL
    result[left] = torch.exp(-0.5*alphaLL*alphaLL) * torch.pow(fact1L * fact2L, -nLL)

    nRR = nR[right]
    tR = t[right]
    alphaRR = alphaR[right]
    fact1R = alphaRR/nRR
    fact2R = nRR/alphaRR - alphaRR + tR
    result[right] = torch.exp(-0.5*alphaRR*alphaRR) * torch.pow(fact1R * fact2R, -nRR)

    norm = double_crystalball_norm(mu, sigma, alphaL, nL, alphaR, nR)
    result = result/norm

    small = result < minval
    result[small] = minval

    return result

'''
normalization
'''
def double_crystalball_norm(mu, sigma, alphaL, nL, alphaR, nR):
    LN_top = torch.exp(-0.5*torch.square(alphaL))*nL
    LN_bottom = alphaL*(nL-1)

    RN_top = torch.exp(-0.5*torch.square(alphaR))*nR
    RN_bottom = alphaR*(nR-1)

    CN = sqrtPiOn2 * (torch.erf(alphaL/sqrt2) + torch.erf(alphaR/sqrt2))

    return (LN_top/LN_bottom + RN_top/RN_bottom + CN) * sigma



'''
Better way of doing things

Constructs semiparametric loss functions with appropriate activations etc
'''

def identity(x):
    return x

def get_sigmoid(threshold):
    return lambda x : (torch.sigmoid(x)-0.5) * 2 * threshold

def get_pos(minval):
    return lambda x : softplus(x) + minval

def general_semiparam(pred, mufun = identity, sigmafun = get_pos(0), alphafun = get_pos(0), nfun = get_pos(1), fixedmu=None):
    i=0
    if fixedmu is None:
        mu = mufun(pred[:,i])
        i+=1
    else:
        mu = fixedmu
    
    sigma = sigmafun(pred[:,i])
    i+=1
    alphaL = alphafun(pred[:,i])
    i+=1
    nL = nfun(pred[:,i])
    i+=1
    alphaR = alphafun(pred[:,i])
    i+=1
    nR = nfun(pred[:,i])

    return mu, sigma, alphaL, nL, alphaR, nR

def general_loss(pred, target, semiparam, fixedmu=None, weight=None):
    batch_size = pred.size()[0]

    param = semiparam(pred, fixedmu)
    prob = smarter(target, *param)
    logprob = torch.log(prob)
    loss = -torch.sum(logprob)/batch_size
    
    return loss

def get_specific_loss(threshold=None, minalpha=0, minn=1, epsilon=1e-6):
    if threshold is not None:
        mufun = get_sigmoid(threshold)
    else:
        mufun = identity

    alphafun = get_pos(minalpha+epsilon)
    nfun = get_pos(minn+epsilon)
    sigmafun = get_pos(epsilon)

    semiparam = lambda x, fixedmu=None: general_semiparam(x, mufun, sigmafun, alphafun, nfun, fixedmu)
    loss = lambda pred, target, fixedmu=None, weight=None : general_loss(pred, target, semiparam, fixedmu, weight)

    return semiparam, loss
