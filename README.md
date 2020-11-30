# graph-met

Attempting to regress MET from PF candidate using a graph neural network. 

Code developed based on examples at: https://github.com/cs230-stanford/cs230-code-examples/tree/master/pytorch.

## Prerequisites 

<pre>
conda install cudatoolkit=10.2
conda install -c pytorch pytorch=1.6.0
export CUDA="cu102"
pip install torch-scatter==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.6.0.html
pip install torch-sparse==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.6.0.html
pip install torch-cluster==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.6.0.html
pip install torch-spline-conv==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.6.0.html
pip install torch-geometric
</pre>

## Input data

Data have been obtained from privately produced NanoAOD, generated with the following package:

https://github.com/mcremone/NanoMET

All PF candidates have been included from the packedPFCandidates collection in MiniAOD:

https://github.com/mcremone/NanoMET/blob/master/python/addPFCands_cff.py#L16-L17

The name of the additional branch in NanoAOD is `JetPFCands`. A dataset corresponding to Z->nunu events has been generated, together with a second dataset composed by a mixture of Drell-Yan and dilepton TT events. The datasets were converted into .npz files using the `data/generate_npz.py` or `data_dytt/generate_npz.py` modules and the output is stored in the `data` or `data_dytt` folder. Each .npz file correspond to one event and contains two arrays:

<pre>
>>> import numpy as np
>>> filename="data_dytt/raw/dy0_event1443.npz"
>>> npzfile = np.load(filename)
>>> npzfile.files
['arr_0', 'arr_1']
>>> npzfile['arr_0']
array([[ 0.62988281, -1.42919922,  0.58825684, ...,  1.        ,
         1.        ,  0.        ],
       [ 0.72998047,  1.87817383, -0.78234863, ..., -1.        ,
         1.        ,  0.        ],
       [ 0.53173828, -1.74511719, -1.13037109, ...,  1.        ,
         1.        ,  0.        ],
       ...,
       [ 0.88671875, -5.07519531,  1.04638672, ...,  0.        ,
         7.        ,  0.        ],
       [ 1.484375  , -5.07519531,  1.74633789, ...,  0.        ,
         7.        ,  0.        ],
       [ 1.44921875, -5.07519531,  1.39648438, ...,  0.        ,
         7.        ,  0.        ]])
>>> npzfile['arr_1']
array([ 17.75378418, -10.78764725,   2.42605209, -34.80409956,
        30.25994253, -31.46089125,  30.17247438, -23.83644676,
        24.76661873, -16.76689124])

</pre>    

The first array, `arr_0`, contains the input features for each PF candidate in the event, that correspond to the sub-arrays. The components of each sub-array correspond to the PF candidate pT, eta, phi, mass, d0, dz, pdgId, charge, pvAssocQuality, and puppiWeight (https://github.com/mcremone/graph-met/blob/clean/data/generate_npz.py#L27-L38). 

The second array, `arr_1`, contains the target values for the event. The values correspond to the px and py of genMET, PF, PUPPI, and DeepMET (https://github.com/mcremone/graph-met/blob/clean/data_dytt/generate_npz.py#L41-L52), or recoil calculated with those (https://github.com/mcremone/graph-met/blob/clean/data/generate_npz.py#L10-L21).

The .npz inputs need to be converted into .pt files to be used for training. From terminal, launch python, then do:

<pre>
import os
from model.data_loader import METDataset
dataset = METDataset(os.environ['PWD']+'/data/')
</pre>   

where `/data/` is the folder where the .npz files used for conversion are. The conversion will generate a sub-folder called `processed` inside such folder, where the .pt files are going to be stored.

## Hardware

To perform the training, a GPU available at the KISTI supercomputer is used. Here are the spec:
<pre>
ssh -Y cms-gpu01
[matteoc@cms-gpu01 ~]$ nvidia-smi 
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 440.33.01    Driver Version: 440.33.01    CUDA Version: 10.2     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  Tesla P100-PCIE...  Off  | 00000000:03:00.0 Off |                    0 |
| N/A   29C    P0    28W / 250W |      0MiB / 16280MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+

</pre>                                                                 

## Training

To launch the training, the `train.py` can be used:


<pre>
python train.py --data data_dytt --ckpts ckpts_dytt --restore_file past
</pre>

The `--data` options indicates the folder where the input data to be used for the training are stored. The `--ckpts` options indicates the folder where the checkpoints produced in the training will be stored, together with evaluation metrics. If the `--data` and `--ckpts` options will not be prvided, defaults are going to be applied. The `--restore_file` options is completely optional and can be used to restart the training from a saved checkpoint (either `last` or `best`).
