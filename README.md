# graph-met

Attempting to regress MET from PF candidate using a Dynamic Reduction Network (https://arxiv.org/abs/2003.08013). 

## Input data

Data have been obtained from privately produced NanoAOD, generated with the following package:

https://github.com/mcremone/NanoMET

All PF candidates have been included from the packedPFCandidates collection in MiniAOD:

https://github.com/mcremone/NanoMET/blob/master/python/addPFCands_cff.py#L16-L17

The name of the additional branch in NanoAOD is `JetPFCands`.

The files that have been used for input generation are:

```/store/mc/RunIISummer19UL18MiniAOD/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/MINIAODSIM/106X_upgrade2018_realistic_v11_L1v1-v2/270000/FCD07DC9-56B4-0A4D-9D0C-4B659C058823.root```
```/store/mc/RunIISummer19UL18MiniAOD/DYToMuMu_pomflux_Pt-30_TuneCP5_13TeV-pythia8/MINIAODSIM/106X_upgrade2018_realistic_v11_L1v1-v2/280000/3063BD33-427C-274C-8ED6-0586930088F8.root```

55k events have been produced for both.

## Hardware

To perform the training, a GPU available at the KISTI supercomputer is used. Here are the spec:
<pre>
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 418.39       Driver Version: 418.39       CUDA Version: 10.1     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  Tesla P100-PCIE...  Off  | 00000000:03:00.0 Off |                    0 |
| N/A   29C    P0    25W / 250W |      0MiB / 16280MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
</pre>                                                                 


