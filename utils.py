import lz4.frame
import cloudpickle
import json
import os.path as osp
import os

# import shutil

import numpy as np
import yaml
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import mplhep as hep

plt.style.use(hep.style.CMS)

import torch

# from torch_cluster import radius


class RunningAverage:
    """A simple class that maintains the running average of a quantity

    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """

    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, value):
        self.total += value
        self.steps += 1

    def __call__(self):
        return self.total / float(self.steps)


def load(filename):
    """Load a coffea file from disk"""
    with lz4.frame.open(filename) as fin:
        output = cloudpickle.load(fin)
    return output


def save(output, filename):
    """Save a coffea object or collection thereof to disk
    This function can accept any picklable object.  Suggested suffix: ``.coffea``
    """
    with lz4.frame.open(filename, "wb") as fout:
        thepickle = cloudpickle.dumps(output)
        fout.write(thepickle)


def save_dict_to_json(d, json_path):
    """Saves dict of floats in json file
    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, "w") as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)


def save_checkpoint(state, is_best, checkpoint):
    """Saves model and training parameters at checkpoint + 'last.pth.tar'. If is_best==True, also saves
    checkpoint + 'best.pth.tar'
    Args:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
        is_best: (bool) True if it is the best model seen till now
        checkpoint: (string) folder where parameters are to be saved
    """
    # ckpt = 'ckpt_{0}.pth.tar'.format(state['epoch'])
    # filepath = osp.join(checkpoint, ckpt)
    filepath = osp.join(checkpoint, "last.pth.tar")
    if is_best:
        filepath = osp.join(checkpoint, "best.pth.tar")
    if not os.path.exists(checkpoint):
        print(
            "Checkpoint Directory does not exist! Making directory {}".format(
                checkpoint
            )
        )
        os.mkdir(checkpoint)
    else:
        print("Checkpoint Directory exists! ")
    torch.save(state, filepath)
    # if is_best:
    #    shutil.copyfile(filepath, osp.join(checkpoint, 'best.pth.tar'))


def load_checkpoint(checkpoint, model, optimizer=None, scheduler=None):
    """Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.
    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """
    if not os.path.exists(checkpoint):
        raise ("File doesn't exist {}".format(checkpoint))
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint["state_dict"])

    if optimizer:
        optimizer.load_state_dict(checkpoint["optim_dict"])

    if scheduler:
        scheduler.load_state_dict(checkpoint["sched_dict"])

    return checkpoint


class info_handler:
    """
    A class that handles usefule information, i.e.
        - loss function values
        - Hyperparameters

    """

    def __init__(self, config):
        with open("configs/" + config + ".yaml", "r") as file:
            self.infos = yaml.load(file, yaml.FullLoader)
        self.infos["output"] = {}
        self.infos["output"]["epoch"] = []
        self.infos["output"]["train_loss"] = []
        self.infos["output"]["eval_loss"] = []
        self.infos["output"]["train_time_per_epoch"] = []
        self.infos["output"]["eval_time_per_epoch"] = []

    def add_epoch(self, epoch, train_loss, eval_loss, train_time, eval_time):
        self.infos["output"]["epoch"].append(epoch)
        self.infos["output"]["train_loss"].append(train_loss)
        self.infos["output"]["eval_loss"].append(eval_loss)
        self.infos["output"]["train_time_per_epoch"].append(train_time)
        self.infos["output"]["eval_time_per_epoch"].append(eval_time)

    def add_info(self, info, val):
        self.infos["output"][info] = val

    def get_info(self, info):
        return self.infos[info]

    def print(self):
        print(self.infos)

    def plot_loss(self, checkpoint):
        plt.plot(
            self.infos["output"]["epoch"],
            self.infos["output"]["train_loss"],
            label="training loss",
        )
        plt.plot(
            self.infos["output"]["epoch"],
            self.infos["output"]["eval_loss"],
            label="evaluation loss",
        )
        plt.legend()
        plt.savefig(checkpoint + "/loss.pdf")
        plt.close()

    def save_infos(self, checkpoint):
        with open(checkpoint + "/output.yaml", "w") as yaml_file:
            yaml.dump(self.infos, yaml_file, default_flow_style=False)


def medians_from_scatter(x, y, x_range=None, n_bins=30):
    if x_range is None:
        left_lim = np.amin(x)
        right_lim = np.amax(x)
    else:
        left_lim = x_range[0]
        right_lim = x_range[1]
    right_lim += (right_lim - left_lim)/100000.
    bin_edges = np.logspace(left_lim, right_lim, n_bins+1)
    bin_centers = (bin_edges[1:] + bin_edges[:-1])/2.
    idxs = [np.logical_and(x >= bin_edges[i], x < bin_edges[i+1]) for i in np.arange(n_bins)]
    median = np.array([np.nanquantile(y[idx], 0.5) for idx in idxs])
    lower = np.array([np.nanquantile(y[idx], 0.16) for idx in idxs])
    upper = np.array([np.nanquantile(y[idx], 0.84) for idx in idxs])
    
    return bin_centers, median, lower, upper


def plot_features(features, model_dir, out):
    plt.hist2d(features[features.pdgId==22]["graphMET_weight"], features[features.pdgId==22]["puppi_weight"], bins=[40,40],range=[[0,1],[0,1]], norm=LogNorm())
    plt.colorbar()
    plt.ylabel("puppi weight")
    plt.xlabel("graphMET weight")
    plt.title("photons")
    plt.savefig(model_dir + "/" + out + "photons-puppi_vs_graphMET_weight.png")
    plt.close()

    plt.hist2d(features[features.pdgId==22]["graphMET_weight"], features[features.pdgId==22]["pT"], bins=[40,40],range=[[0.01,100],[0,1]], norm=LogNorm())
    plt.colorbar()
    plt.ylabel("pT")
    plt.yscale("log")
    plt.xlabel("graphMET weight")
    plt.title("photons")
    plt.savefig(model_dir + "/" + out + "photons-pT_vs_graphMET_weight.png")
    plt.close()

    plt.hist2d(features[features.pdgId==130]["graphMET_weight"], features[features.pdgId==130]["puppi_weight"], bins=[40,40],range=[[0,1],[0,1]], norm=LogNorm())
    plt.colorbar()
    plt.ylabel("puppi weight")
    plt.xlabel("graphMET weight")
    plt.title("neutrals")
    plt.savefig(model_dir + "/" + out + "neutrals-puppi_vs_graphMET_weight.png")
    plt.close()

    plt.hist2d(features[features.pdgId==130]["graphMET_weight"], features[features.pdgId==130]["pT"], bins=[40,40],range=[[0.01,100],[0,1]], norm=LogNorm())
    plt.colorbar()
    plt.ylabel("pT")
    plt.yscale("log")
    plt.xlabel("graphMET weight")
    plt.title("neutrals")
    plt.savefig(model_dir + "/" + out + "neutrals-pT_vs_graphMET_weight.png")
    plt.close()

    analytic_22, median_22, low_22, up_22 = medians_from_scatter(features[features.pdgId==22]["pT"], features[features.pdgId==22]["graphMET_weight"], x_range=(0.1,10), n_bins=20)
    plt.plot(analytic_22, median_22, alpha=0.7, color="blue", label="photons")
    plt.fill_between(analytic_22, low_22, up_22, facecolor="blue", alpha=0.2)
    analytic_130, median_130, low_130, up_130 = medians_from_scatter(features[features.pdgId==130]["pT"], features[features.pdgId==130]["graphMET_weight"], x_range=(0.1,10), n_bins=20)
    plt.plot(analytic_130, median_130, alpha=0.7, color="green", label="neutrals")
    plt.fill_between(analytic_130, low_130, up_130, facecolor="green", alpha=0.2)
    plt.xlabel("PF pT")
    plt.xlim([0.1,20])
    plt.xscale("log")
    plt.ylabel("graphMET weight")
    plt.ylim([0,1])
    plt.legend()
    plt.savefig(model_dir + "/" + out + "particles-pT_vs_graphMET_weight.png")
    plt.close()

    # analytic_PV0, median_PV0, low_PV0, up_PV0 = medians_from_scatter(features[features.fromPV==0]["pT"], features[features.fromPV==0]["graphMET_weight"], x_range=(0.1,10), n_bins=20)
    # plt.plot(analytic_PV0, median_PV0, alpha=0.7, color="blue", label="fromPV=0")
    # plt.fill_between(analytic_PV0, low_PV0, up_PV0, facecolor="blue", alpha=0.2)
    # analytic_PV1, median_PV1, low_PV1, up_PV1 = medians_from_scatter(features[features.fromPV==1]["pT"], features[features.fromPV==1]["graphMET_weight"], x_range=(0.1,10), n_bins=20)
    # plt.plot(analytic_PV1, median_PV1, alpha=0.7, color="green", label="fromPV=1")
    # plt.fill_between(analytic_PV1, low_PV1, up_PV1, facecolor="green", alpha=0.2)
    # analytic_PV2, median_PV2, low_PV2, up_PV2 = medians_from_scatter(features[features.fromPV==2]["pT"], features[features.fromPV==2]["graphMET_weight"], x_range=(0.1,10), n_bins=20)
    # plt.plot(analytic_PV2, median_PV2, alpha=0.7, color="red", label="fromPV=2")
    # plt.fill_between(analytic_PV2, low_PV2, up_PV2, facecolor="red", alpha=0.2)
    # analytic_PV3, median_PV3, low_PV3, up_PV3 = medians_from_scatter(features[features.fromPV==3]["pT"], features[features.fromPV==3]["graphMET_weight"], x_range=(0.1,10), n_bins=20)
    # plt.plot(analytic_PV3, median_PV3, alpha=0.7, color="black", label="fromPV=3")
    # plt.fill_between(analytic_PV3, low_PV3, up_PV3, facecolor="black", alpha=0.2)
    # plt.xlabel("PF pT")
    # plt.xlim([0.01,100])
    # plt.xscale("log")
    # plt.ylabel("graphMET weight")
    # plt.ylim([0,1])
    # plt.legend()
    # plt.savefig(model_dir + "/" + out + "fromPV-pT_vs_graphMET_weight.png")
    # plt.close()


    # plt.plot(features[features.pdgId==22]["graphMET_weight"], features[features.pdgId==22]["puppiWeight"], ".b", label="photons")
    # plt.plot(features[features.pdgId==130]["graphMET_weight"], features[features.pdgId==130]["puppiWeight"], ".g", label="neutrals")
    # plt.ylabel("puppi weight")
    # plt.ylim([0,1])
    # plt.xlabel("graphMET weight")
    # plt.xlim([0,1])
    # plt.legend()
    # plt.savefig(model_dir + "/" + out + "particles-puppi_vs_graphMET_weight.png")
    # plt.close()

    # plt.plot(features[features.pdgId==22]["graphMET_weight"], features[features.pdgId==22]["pT"], ".b", label="photons")
    # plt.plot(features[features.pdgId==130]["graphMET_weight"], features[features.pdgId==130]["pT"], ".g", label="neutrals")
    # plt.ylabel("pT")
    # plt.ylim([0.01,100])
    # plt.yscale("log")
    # plt.xlabel("graphMET weight")
    # plt.xlim([0,1])
    # plt.legend()
    # plt.savefig(model_dir + "/" + out + "particles-pT_vs_graphMET_weight.png")
    # plt.close()

    # plt.plot(features[features.fromPV==0]["graphMET_weight"], features[features.fromPV==0]["pT"], ".b", label="fromPV=0")
    # plt.plot(features[features.fromPV==1]["graphMET_weight"], features[features.fromPV==1]["pT"], ".g", label="fromPV=1")
    # plt.plot(features[features.fromPV==2]["graphMET_weight"], features[features.fromPV==2]["pT"], ".r", label="fromPV=2")
    # plt.plot(features[features.fromPV==3]["graphMET_weight"], features[features.fromPV==3]["pT"], ".k", label="fromPV=3")
    # plt.ylabel("pT")
    # plt.ylim([0.01,100])
    # plt.yscale("log")
    # plt.xlabel("graphMET weight")
    # plt.xlim([0,1])
    # plt.legend()
    # plt.savefig(model_dir + "/" + out + "fromPV-pT_vs_graphMET_weight.png")
    # plt.close()

    # counts, x_edges, y_edges = np.histogram2d(features[features.fromPV==0]["graphMET_weight"], features[features.fromPV==0]["pT"], bins=40)
    # #plt.hist2d(features[features.fromPV==0]["graphMET_weight"], features[features.fromPV==0]["pT"], bins=[40,40],range=[[0.01,100],[0,1]], norm=LogNorm())
    # fig, ax = plt.subplots()
    # ax.pcolormesh(x_edges, y_edges, counts.T)
    # ax.set_xscale("log")
    # #fig.colorbar()
    # #fig.ylabel("pT")
    # #plt.yscale("log")
    # #fig.xlabel("graphMET weight")
    # #fig.title("fromPV = 0")
    # fig.savefig(model_dir + "/" + out + "PV0-pT_vs_graphMET_weight.png")
    # plt.close()

    plt.hist(features[features.fromPV==0]["graphMET_weight"], bins=40, density=True, histtype="step", label="fromPV=0")
    plt.hist(features[features.fromPV==1]["graphMET_weight"], bins=40, density=True, histtype="step", label="fromPV=1")
    plt.hist(features[features.fromPV==2]["graphMET_weight"], bins=40, density=True, histtype="step", label="fromPV=2")
    plt.hist(features[features.fromPV==3]["graphMET_weight"], bins=40, density=True, histtype="step", label="fromPV=3")
    plt.ylabel("normalized")
    plt.yscale("log")
    plt.xlabel("graphMET weight")
    plt.legend()
    plt.savefig(model_dir + "/" + out + "fromPV-graphMET_weight.png")
    plt.close()

    plt.hist(features[(features.puppi_weight==0) & (features.charge!=0)]["graphMET_weight"], bins=40, density=True, histtype="step", label="puppi=0")
    plt.hist(features[(features.puppi_weight==1) & (features.charge!=0)]["graphMET_weight"], bins=40, density=True, histtype="step", label="puppi=1")
    plt.ylabel("normalized")
    plt.yscale("log")
    plt.xlabel("graphMET weight")
    plt.legend()
    plt.savefig(model_dir + "/" + out + "puppi-graphMET_weight.png")
    plt.close()
