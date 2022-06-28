"""Evaluates the model"""

import argparse
import os.path as osp
import os
import time
from tqdm import tqdm

import numpy as np
import pandas as pd
import json
import torch

import utils
import add_eval_plots
import model.net as net
import model.data_loader as data_loader
from model.nano_loader import METDataset

from torch_cluster import radius_graph, knn_graph

import matplotlib.pyplot as plt
import mplhep as hep
plt.style.use(hep.style.CMS)

parser = argparse.ArgumentParser()
parser.add_argument("--restore_file", default="best", help="Option to restore a certain model. There are two possibilities 'best' and 'last'. The used model for restoration is defied in the used config.yaml")
parser.add_argument("--config", default="config.yaml", help="Name of the config taht should be used. This is a yaml file that includes all the ajustable parameters for the network training/evaluation.")
parser.add_argument("--out", default="", help="Additional string that can be defined to differentiate result plots for different data sets.")
parser.add_argument('--addPlots', default=False, help="Option to produce additional plots during the evaluation process.")

def evaluate(model, device, loss_fn, dataloader, metrics, net_info, model_dir, out="", addPlots=False):
    """Evaluate a neural network model.

    Args:
        model: the neural network
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: data that should be evaluated
        metrics: a dictionary of functions that compute a metric using the output and labels of each batch
    """
    # set model to evaluation mode
    model.eval()

    with torch.no_grad():
        # predefinitions for event loop
        loss_avg_arr = []
        qT_arr = []
        has_deepmet = False
        resolutions_arr = {
            "graphMET":      [[],[],[]],
            "pfMET":    [[],[],[]],
            "puppiMET": [[],[],[]],
        }

        colors = {
            "pfMET": "black",
            "puppiMET": "red",
            "deepMETResponse": "blue",
            "deepMETResolution": "green",
            "graphMET": "magenta",
        }

        labels = {
            "pfMET": "PF MET",
            "puppiMET": "PUPPI MET",
            "deepMETResponse": "DeepMETResponse",
            "deepMETResolution": "DeepMETResolution",
            "graphMET": "Graph MET"
        }

        if addPlots:
            features = {
                "graphMET_weight": [],
                "pT": [],
                "eta": [],
                "fromPV": [],
                "charge": [],
                "pdgId": [],
                "puppi_weight": [],
            }

        start_time = time.time()
        # compute metrics over the dataset
        with tqdm(total=len(dataloader)) as t:
            for data in dataloader:

                has_deepmet = (data.y.size()[1] > 6)
                
                if has_deepmet == True and "deepMETResponse" not in resolutions_arr.keys():
                    resolutions_arr.update({
                        "deepMETResponse": [[],[],[]],
                        "deepMETResolution": [[],[],[]]
                    })
                
                data = data.to(device)

                if net_info["graph"]["calculation"] == "static":
                    if net_info["graph"]["type"] == "radius_graph":
                        # calculate phi from momenta pY and pX
                        phi = torch.atan2(data.x[:, 1], data.x[:, 0])
                        # define eta-phi plane for graph
                        etaphi = torch.cat([data.x[:, 3][:, None], phi[:, None]], dim=1)
                        # calculate connections between PF candidates based on a radius
                        edge_index = radius_graph(
                            etaphi,
                            r=net_info["graph"]["deltaR"],
                            batch=data.batch,
                            loop=net_info["graph"]["loop"],
                            max_num_neighbors=net_info["graph"]["max_num_neighbors"],
                        )
                    elif net_info["graph"]["type"] == "knn_graph":
                        # calculate phi from momenta pY and pX
                        phi = torch.atan2(data.x[:, 1], data.x[:, 0])
                        # define eta-phi plane for graph
                        etaphi = torch.cat([data.x[:, 3][:, None], phi[:, None]], dim=1)
                        # calculate connections between PF candidates based on k nearest neighbors
                        edge_index = knn_graph(
                                etaphi,
                                k=net_info["graph"]["k"],
                                batch=data.batch,
                                loop=net_info["graph"]["loop"],
                        )
                    else:
                        print("Wrong graph type!")

                elif net_info["graph"]["calculation"] == "dynamic":
                    edge_index = None
                else:
                    print("Graph not defined.")

                result = model(data.x, edge_index=edge_index, batch=data.batch)
                loss = loss_fn(result, data.x, data.y, data.batch, net_info["loss"])

                if addPlots:
                    features["graphMET_weight"] = np.concatenate((features["graphMET_weight"],result.cpu().detach().numpy()))
                    features["pT"] = np.concatenate((features["pT"],data.x[:, 2].cpu().detach().numpy()))
                    features["eta"] = np.concatenate((features["eta"],data.x[:, 3].cpu().detach().numpy()))
                    features["pdgId"] = np.concatenate((features["pdgId"],data.x[:, 8].cpu().detach().numpy()))
                    features["charge"] = np.concatenate((features["charge"],data.x[:, 9].cpu().detach().numpy()))
                    features["fromPV"] = np.concatenate((features["fromPV"],data.x[:, 10].cpu().detach().numpy()))
                    features["puppi_weight"] = np.concatenate((features["puppi_weight"],data.x[:, 7].cpu().detach().numpy()))

                # compute all metrics on this batch
                resolutions, qT = metrics["resolution"](result, data.x, data.y, data.batch)
                for key in resolutions_arr:
                    for i in range(len(resolutions_arr[key])):
                        resolutions_arr[key][i]=np.concatenate((resolutions_arr[key][i],resolutions[key][i]))
                qT_arr=np.concatenate((qT_arr,qT))
                loss_avg_arr.append(loss.cpu().item())
                
                t.update()

        evaluation_time = time.time() - start_time
        # compute mean of all metrics in summary
        max_x=400 # max qT value
        n_x=25 # number of bins

        bin_edges=np.arange(0, max_x, int(max_x/n_x))
        inds=np.digitize(qT_arr,bin_edges)
        qT_hist=[]
        for i in range(1, len(bin_edges)):
            qT_hist.append((bin_edges[i]+bin_edges[i-1])/2.)
        
        resolution_hists={}
        for key in resolutions_arr:

            R_arr=resolutions_arr[key][2] 
            u_perp_arr=resolutions_arr[key][0]
            u_par_arr=resolutions_arr[key][1]

            u_perp_hist=[]
            u_perp_scaled_hist=[]
            u_par_hist=[]
            u_par_scaled_hist=[]
            R_hist=[]

            for i in range(1, len(bin_edges)):
                R_i=R_arr[np.where(inds==i)[0]]
                if R_i.size == 0:
                    R_mean = 0.
                else:
                    R_mean = np.mean(R_i)
                R_hist.append(R_mean)
                u_perp_i=u_perp_arr[np.where(inds==i)[0]]
                u_perp_scaled_i=u_perp_i/R_mean
                if u_perp_i.size==0:
                    u_perp_hist.append(0.)
                    u_perp_scaled_hist.append(0.)
                else:
                    u_perp_hist.append(np.quantile(u_perp_i,0.68))
                    u_perp_scaled_hist.append(np.quantile(u_perp_scaled_i,0.68))
                
                u_par_i=u_par_arr[np.where(inds==i)[0]]
                u_par_scaled_i=u_par_i/R_mean
                if u_par_i.size==0:
                    u_par_hist.append(0.)
                    u_par_scaled_hist.append(0.)
                else:
                    u_par_hist.append((np.quantile(u_par_i,0.84)-np.quantile(u_par_i,0.16))/2.)
                    u_par_scaled_hist.append((np.quantile(u_par_scaled_i,0.84)-np.quantile(u_par_scaled_i,0.16))/2.)

            u_perp_resolution=np.histogram(qT_hist, bins=n_x, range=(0,max_x), weights=u_perp_hist)
            u_perp_scaled_resolution=np.histogram(qT_hist, bins=n_x, range=(0,max_x), weights=u_perp_scaled_hist)
            u_par_resolution=np.histogram(qT_hist, bins=n_x, range=(0,max_x), weights=u_par_hist)
            u_par_scaled_resolution=np.histogram(qT_hist, bins=n_x, range=(0,max_x), weights=u_par_scaled_hist)
            R=np.histogram(qT_hist, bins=n_x, range=(0,max_x), weights=R_hist)

            plt.figure()
            plt.figure(1)
            plt.plot(qT_hist, u_perp_hist,        color=colors[key], label=labels[key])
            plt.figure(2)
            plt.plot(qT_hist, u_perp_scaled_hist, color=colors[key], label=labels[key])
            plt.figure(3)
            plt.plot(qT_hist, u_par_hist,         color=colors[key], label=labels[key])
            plt.figure(4)
            plt.plot(qT_hist, u_par_scaled_hist,  color=colors[key], label=labels[key])
            plt.figure(5)
            plt.plot(qT_hist, R_hist,             color=colors[key], label=labels[key])
                

            resolution_hists[key] = {
                "u_perp_resolution": u_perp_resolution,
                "u_perp_scaled_resolution": u_perp_scaled_resolution,
                "u_par_resolution": u_par_resolution,
                "u_par_scaled_resolution": u_par_scaled_resolution,
                "R": R
            }

        plt.figure(1)
        plt.axis([0, 400, 0, 45])
        plt.xlabel(r'$q_{T}$ [GeV]')
        plt.ylabel(r'$\sigma (u_{\perp})$ [GeV]')
        plt.legend()
        plt.savefig(model_dir+"/"+out+"resol_perp.png")
        plt.clf()
        plt.close()

        plt.figure(2)
        plt.axis([0, 400, 0, 45])
        plt.xlabel(r'$q_{T}$ [GeV]')
        plt.ylabel(r'Scaled $\sigma (u_{\perp})$ [GeV]')
        plt.legend()
        plt.savefig(model_dir+"/"+out+"resol_perp_scaled.png")
        plt.clf()
        plt.close()

        plt.figure(3)
        plt.axis([0, 400, 0,60])
        plt.xlabel(r'$q_{T}$ [GeV]')
        plt.ylabel(r'$\sigma (u_{\parallel})$ [GeV]')
        plt.legend()
        plt.savefig(model_dir+"/"+out+"resol_parallel.png")
        plt.clf()
        plt.close()

        plt.figure(4)
        plt.axis([0, 400, 0, 60])
        plt.xlabel(r'$q_{T}$ [GeV]')
        plt.ylabel(r'Scaled $\sigma (u_{\parallel})$ [GeV]')
        plt.legend()
        plt.savefig(model_dir+"/"+out+"resol_parallel_scaled.png")
        plt.clf()
        plt.close()

        plt.figure(5)
        plt.axis([0, 400, 0, 1.2])
        plt.axhline(y=1.0, color="black", linestyle="-.")
        plt.xlabel(r'$q_{T}$ [GeV]')
        plt.ylabel(r'Response $-\frac{<u_{\parallel}>}{<q_{T}>}$')
        plt.legend()
        plt.savefig(model_dir+"/"+out+"response_parallel.png")
        plt.clf()
        plt.close()

        if addPlots:
            add_eval_plots.plot_features(pd.DataFrame.from_dict(features), model_dir, out, eta_range=(0,5))
            add_eval_plots.plot_features(pd.DataFrame.from_dict(features), model_dir, out, eta_range=(0,1.3))
            add_eval_plots.plot_features(pd.DataFrame.from_dict(features), model_dir, out, eta_range=(1.3,2))
            add_eval_plots.plot_features(pd.DataFrame.from_dict(features), model_dir, out, eta_range=(2,2.5))
            add_eval_plots.plot_features(pd.DataFrame.from_dict(features), model_dir, out, eta_range=(2.5,3))
            add_eval_plots.plot_features(pd.DataFrame.from_dict(features), model_dir, out, eta_range=(3,5))

        metrics_mean = {
            "loss": np.mean(loss_avg_arr),
        }
        metrics_string = " ; ".join("{}: {:05.3f}".format(k, v)
                                    for k, v in metrics_mean.items())
        print("- Eval metrics : " + metrics_string)
        return metrics_mean, resolution_hists, evaluation_time


if __name__ == "__main__":
    """
        Evaluate the model on the test set.
    """
    # Load the parameters
    args = parser.parse_args()

    info_handler = utils.info_handler(args.config, args.restore_file)
    data_info = info_handler.get_info("data_info")
    train_info = info_handler.get_info("train_info")
    net_info = info_handler.get_info("net_info")

    if data_info["load"] == "rootfiles":
        dataset = METDataset(
            data_info["dataset"],
            data_info["validation_split"],
            data_info["batch_size"],
            data_info["seed"],
        )
        dataloaders = dataset.get_data()
    elif data_info["load"] == "ptfiles":
        dataloaders = data_loader.fetch_dataloader(data_dir=osp.join(os.environ["PWD"],data_info["dataset"]),
                                                batch_size=data_info["batch_size"],
                                                validation_split=data_info["validation_split"])
    else:
        print("You want to load a wrond data format.")

    test_dl = dataloaders["test"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Evaluation performed on:", device)
    # restriction on gpu memory usage
    torch.cuda.set_per_process_memory_fraction(train_info["gpu_memory_usage"])

    model = net.Net(net_info).to(device)
    utils.print_model_summary(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=train_info["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=train_info["factor"],
        patience=train_info["patience"],
        threshold=train_info["threshold"],
    )

    loss_fn = net.loss_fn
    metrics = net.metrics
    model_dir = osp.join(os.environ["PWD"], data_info["output_dir"])

    # Reload weights from the saved file
    ckpt_file = osp.join(model_dir, args.restore_file + ".pth.tar")
    ckpt = utils.load_checkpoint(ckpt_file, model, optimizer, scheduler)
    epoch = ckpt["epoch"]
    with open(osp.join(model_dir, "metrics_val_best.json")) as restore_metrics:
            best_validation_loss = json.load(restore_metrics)["loss"]

    # Evaluate
    test_metrics, resolutions, evaluation_time = evaluate(model, device, loss_fn, test_dl, metrics, net_info, model_dir, args.out, args.addPlots)
