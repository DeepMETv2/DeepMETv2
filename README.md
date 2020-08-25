# graph-met

Attempting to regress MET from PF candidate using a Dynamic Reduction Network (https://arxiv.org/abs/2003.08013). 

## Input data

Data have been obtained from privately produced NanoAOD, generated with the following package:

https://github.com/mcremone/NanoMET

All PF candidates have been included from the packedPFCandidates collection in MiniAOD:

https://github.com/mcremone/NanoMET/blob/master/python/addPFCands_cff.py#L16-L17

The name of the additional branch in NanoAOD is `JetPFCands`.

## Hardware

To perform the training, a GPU available at the KISTI supercomputer is used. Here are the spec:

+-----------------------------------------------------------------------------+<br />
| NVIDIA-SMI 418.39       Driver Version: 418.39       CUDA Version: 10.1     |<br />
|-------------------------------+----------------------+----------------------+<br />
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |<br />
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |<br />
|===============================+======================+======================|<br />
|   0  Tesla P100-PCIE...  Off  | 00000000:03:00.0 Off |                    0 |<br />
| N/A   29C    P0    25W / 250W |      0MiB / 16280MiB |      0%      Default |<br />
+-------------------------------+----------------------+----------------------+<br />
                                                                               


