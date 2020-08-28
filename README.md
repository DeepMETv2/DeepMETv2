# graph-met

Attempting to regress MET from PF candidate using a Dynamic Reduction Network (https://arxiv.org/abs/2003.08013). 

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

The name of the additional branch in NanoAOD is `JetPFCands`.

The files that have been used for input generation are:

```/store/mc/RunIISummer19UL18MiniAOD/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/MINIAODSIM/106X_upgrade2018_realistic_v11_L1v1-v2/270000/FCD07DC9-56B4-0A4D-9D0C-4B659C058823.root```
```/store/mc/RunIISummer19UL18MiniAOD/DYToMuMu_pomflux_Pt-30_TuneCP5_13TeV-pythia8/MINIAODSIM/106X_upgrade2018_realistic_v11_L1v1-v2/280000/3063BD33-427C-274C-8ED6-0586930088F8.root```

55k events have been produced for both. They were converted into .npz files using the `generate_npz.py` module and the output is stored in the `data` folder. Each .npz file correspond to one event and contains two arrays:

<pre>
>>> npzfile = np.load(filename)
>>> npzfile.files
['arr_0', 'arr_1']
>>> npzfile['arr_0']
array([[ 5.8935547e-01, -2.0834961e+00, -6.3220215e-01,  1.3952637e-01],
       [ 8.8574219e-01,  1.0312500e+00,  2.3701172e+00,  1.3952637e-01],
       [ 9.1601562e-01, -1.3386230e+00, -2.2963867e+00,  1.3952637e-01],
       ...,
       [ 1.1855469e+00, -5.0761719e+00,  1.7465820e+00,  1.3113022e-06],
       [ 2.3291016e-01, -5.0761719e+00,  1.0463867e+00,  2.3841858e-07],
       [ 1.4609375e+00, -5.0771484e+00,  1.3964844e+00,  0.0000000e+00]],
      dtype=float32)
>>> npzfile['arr_1']
array([92.], dtype=float32)
</pre>    

The first array, `arr_0`, contains the 4-vectors for each PF candidate in the event, that correspond to the sub-arrays. The components of each sub-array correspond to the PF candidate pT, eta, phi, and mass. 

The second array, `arr_1`, contains the genMET value for the event.

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


