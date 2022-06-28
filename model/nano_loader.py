# imports
import numpy as np
import uproot
import awkward as ak
import torch
from tqdm import tqdm
import time
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import concurrent.futures


class METDataset:
    """PyTorch geometric dataset loader for MET"""

    def __init__(self, files, features, labels, validation_split, batch_size, random_seed):

        tree_name = "Events"

        start_time = time.time()
        print("-" * 50)
        print("Start loading dataset:", files)
        print("-" * 50)
        executor = concurrent.futures.ThreadPoolExecutor()
        tree = uproot.concatenate(files+":"+tree_name, features+labels,decompression_executor=executor,interpretation_executor=executor)

        tree["PFCands_px"] = tree["PFCands_pt"] * np.cos(tree["PFCands_phi"])
        tree["PFCands_py"] = tree["PFCands_pt"] * np.sin(tree["PFCands_phi"])

        tree["GenMET_px"] = tree["GenMET_pt"] * np.cos(tree["GenMET_phi"])
        tree["GenMET_py"] = tree["GenMET_pt"] * np.sin(tree["GenMET_phi"])
        tree["MET_px"] = tree["MET_pt"] * np.cos(tree["MET_phi"])
        tree["MET_py"] = tree["MET_pt"] * np.sin(tree["MET_phi"])
        tree["PuppiMET_px"] = tree["PuppiMET_pt"] * np.cos(tree["PuppiMET_phi"])
        tree["PuppiMET_py"] = tree["PuppiMET_pt"] * np.sin(tree["PuppiMET_phi"])
        tree["DeepMETResponseTune_px"] = tree["DeepMETResponseTune_pt"] * np.cos(
            tree["DeepMETResponseTune_phi"]
        )
        tree["DeepMETResponseTune_py"] = tree["DeepMETResponseTune_pt"] * np.sin(
            tree["DeepMETResponseTune_phi"]
        )
        tree["DeepMETResolutionTune_px"] = tree["DeepMETResolutionTune_pt"] * np.cos(
            tree["DeepMETResolutionTune_phi"]
        )
        tree["DeepMETResolutionTune_py"] = tree["DeepMETResolutionTune_pt"] * np.sin(
            tree["DeepMETResolutionTune_phi"]
        )

        n_events = len(tree)
        dataset = []
        print("Looping over all events in dataset.")
        with tqdm(total=n_events) as t:
            for i in range(n_events):
                pX = ak.to_numpy(tree["PFCands_px"][i]).astype(np.float32)
                pY = ak.to_numpy(tree["PFCands_py"][i]).astype(np.float32)
                pT = ak.to_numpy(tree["PFCands_pt"][i]).astype(np.float32)
                eta = ak.to_numpy(tree["PFCands_eta"][i]).astype(np.float32)
                d0 = ak.to_numpy(tree["PFCands_d0"][i]).astype(np.float32)
                dz = ak.to_numpy(tree["PFCands_dz"][i]).astype(np.float32)
                mass = ak.to_numpy(tree["PFCands_mass"][i]).astype(np.float32)
                puppiWeight = ak.to_numpy(tree["PFCands_puppiWeight"][i]).astype(
                    np.float32
                )
                pdgId = ak.to_numpy(tree["PFCands_pdgId"][i]).astype(np.float32)
                charge = ak.to_numpy(tree["PFCands_charge"][i]).astype(np.float32)
                fromPV = ak.to_numpy(tree["PFCands_fromPV"][i]).astype(np.float32)

                # puppi weights should be the last continuous feature, so it can be remove/added easier later in the training 
                x = np.stack(
                    (pX, pY, pT, eta, d0, dz, mass, puppiWeight, pdgId, charge, fromPV),
                    axis=-1,
                )
                x = np.nan_to_num(x, nan=0.0, posinf=5000.0, neginf=-5000.0)

                metX_true = ak.to_numpy(tree["GenMET_px"][i]).astype(np.float32)
                metY_true = ak.to_numpy(tree["GenMET_py"][i]).astype(np.float32)
                metX_pf = ak.to_numpy(tree["MET_px"][i]).astype(np.float32)
                metY_pf = ak.to_numpy(tree["MET_py"][i]).astype(np.float32)
                metX_puppi = ak.to_numpy(tree["PuppiMET_px"][i]).astype(np.float32)
                metY_puppi = ak.to_numpy(tree["PuppiMET_py"][i]).astype(np.float32)
                metX_deepMETResp = ak.to_numpy(
                    tree["DeepMETResponseTune_px"][i]
                ).astype(np.float32)
                metY_deepMETResp = ak.to_numpy(
                    tree["DeepMETResponseTune_py"][i]
                ).astype(np.float32)
                metX_deepMETReso = ak.to_numpy(
                    tree["DeepMETResolutionTune_px"][i]
                ).astype(np.float32)
                metY_deepMETReso = ak.to_numpy(
                    tree["DeepMETResolutionTune_py"][i]
                ).astype(np.float32)

                y = np.column_stack(
                    [
                        metX_true,
                        metY_true,
                        metX_pf,
                        metY_pf,
                        metX_puppi,
                        metY_puppi,
                        metX_deepMETResp,
                        metY_deepMETResp,
                        metX_deepMETReso,
                        metY_deepMETReso,
                    ]
                )
                edge_index = torch.empty((2, 0), dtype=torch.long)

                event_data = Data(
                    x=torch.from_numpy(x), edge_index=edge_index, y=torch.from_numpy(y)
                )
                dataset.append(event_data)
                t.update()

        load_time = time.time() - start_time
        print("-" * 50)
        print(
            "Loading dataset finished. Time for loading: {:.2f} sec".format(load_time)
        )
        print("-" * 50)
        dataset_size = len(dataset)
        split = int(np.floor(validation_split * dataset_size))
        train_set, val_set = torch.utils.data.random_split(
            dataset,
            [dataset_size - split, split],
            generator=torch.Generator().manual_seed(random_seed),
        )
        print("Number of training events: {}".format(len(train_set)))
        print("Number of validation events: {}".format(len(val_set)))
        print("-" * 50)
        self.dataloaders = {
            "train": DataLoader(train_set, batch_size=batch_size, shuffle=False),
            "test": DataLoader(val_set, batch_size=batch_size, shuffle=False),
        }

    def get_data(self):
        return self.dataloaders
