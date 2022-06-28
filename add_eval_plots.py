import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm



def binned_median(x, y, nbins=30, bins = None, x_range=None):
    """
    Divide the x axis into sections and return groups of y based on its x value
    """
    if bins is None:
        if x_range is None:
            bins = np.linspace(x.min(), x.max(), nbins)
        else:
            bins = np.linspace(x_range[0], x_range[1], nbins)

    indices = np.digitize(x, bins)
    output = []
    for i in range(0, len(bins)):
        output.append(y[indices==i])
    #
    # prepare a dataframe with cols: median; 1std up, 1std down
    df_names = ["median", "16", "84"]
    df = pd.DataFrame(columns = df_names)
    to_delete = []
    # for each bin, determine the std ranges
    for y_set in output:
        if y_set.size > 0:
            intervals = np.percentile(y_set, q = [50, 16, 84])
            df = df.append(pd.DataFrame([intervals], columns = df_names))
        else:
            # just in case there are no elements in the bin
            to_delete.append(len(df) + len(to_delete))
            
    # add x values
    bins = np.delete(bins, to_delete)
    df["x"] = bins

    return df


def plot_features(features, model_dir, out, eta_range):
    if not os.path.exists(model_dir+"/additional_plots"):
        print(
            "Directory for additional plots does not exist! Making directory {}".format(
                model_dir+"/additional_plots"
            )
        )
        os.mkdir(model_dir+"/additional_plots")

    plot_dir = model_dir+"/additional_plots"

    plt.hist2d(features[(features.pdgId==22) & (features.eta>=eta_range[0]) & (features.eta<eta_range[1])]["puppi_weight"], features[(features.pdgId==22)& (features.eta>=eta_range[0]) & (features.eta<eta_range[1])]["graphMET_weight"], bins=[40,40],range=[[0,1],[0,1]], norm=LogNorm())
    plt.colorbar()
    plt.xlabel("puppi weight")
    plt.ylabel("graphMET weight")
    plt.title("photons, eta = {} - {}".format(str(eta_range[0]),str(eta_range[1])))
    plt.savefig(plot_dir + "/" + out + "photons-puppi_vs_graphMET_weight_eta_{}_{}.png".format(str(eta_range[0]),str(eta_range[1])))
    plt.close()

    fig, ax = plt.subplots(nrows=1, ncols=1)
    xbins = np.logspace( np.log10(0.1),  np.log10(20), 40)
    ybins = np.linspace(0, 1, 40)
    counts, _, _ = np.histogram2d(features[(features.pdgId==22)& (features.eta>=eta_range[0]) & (features.eta<eta_range[1])]["pT"], features[(features.pdgId==22)& (features.eta>=eta_range[0]) & (features.eta<eta_range[1])]["graphMET_weight"], bins=(xbins, ybins))
    pcm = ax.pcolormesh(xbins, ybins, counts, norm=LogNorm())
    plt.colorbar(pcm)
    ax.set_xscale("log")
    ax.set_xlabel("pT")
    ax.set_ylabel("graphMET weight")
    ax.set_title("photons, eta = {} - {}".format(str(eta_range[0]),str(eta_range[1])))
    plt.savefig(plot_dir + "/" + out + "photons-pT_vs_graphMET_weight_eta_{}_{}.png".format(str(eta_range[0]),str(eta_range[1])))
    plt.close()


    plt.hist2d(features[(features.pdgId==130) & (features.eta>=eta_range[0]) & (features.eta<eta_range[1])]["puppi_weight"], features[(features.pdgId==130)& (features.eta>=eta_range[0]) & (features.eta<eta_range[1])]["graphMET_weight"], bins=[40,40],range=[[0,1],[0,1]], norm=LogNorm())
    plt.colorbar()
    plt.xlabel("puppi weight")
    plt.ylabel("graphMET weight")
    plt.title("neutrals, eta = {} - {}".format(str(eta_range[0]),str(eta_range[1])))
    plt.savefig(plot_dir + "/" + out + "neutrals-puppi_vs_graphMET_weight_eta_{}_{}.png".format(str(eta_range[0]),str(eta_range[1])))
    plt.close()

    fig, ax = plt.subplots(nrows=1, ncols=1)
    xbins = np.logspace( np.log10(0.1),  np.log10(20), 40)
    ybins = np.linspace(0, 1, 40)
    counts, _, _ = np.histogram2d(features[(features.pdgId==130)& (features.eta>=eta_range[0]) & (features.eta<eta_range[1])]["pT"], features[(features.pdgId==130)& (features.eta>=eta_range[0]) & (features.eta<eta_range[1])]["graphMET_weight"], bins=(xbins, ybins))
    pcm = ax.pcolormesh(xbins, ybins, counts, norm=LogNorm())
    plt.colorbar(pcm)
    ax.set_xscale("log")
    ax.set_xlabel("pT")
    ax.set_ylabel("graphMET weight")
    ax.set_title("neutrals, eta = {} - {}".format(str(eta_range[0]),str(eta_range[1])))
    plt.savefig(plot_dir + "/" + out + "neutrals-pT_vs_graphMET_weight_eta_{}_{}.png".format(str(eta_range[0]),str(eta_range[1])))
    plt.close()

    df_photons = binned_median(features[(features.pdgId==22)& (features.eta>=eta_range[0]) & (features.eta<eta_range[1])]["puppi_weight"], features[(features.pdgId==22)& (features.eta>=eta_range[0]) & (features.eta<eta_range[1])]["graphMET_weight"], nbins=40, bins = None, x_range=(0,1))
    plt.plot(df_photons.x, df_photons["median"], alpha=0.7, color="blue", label="photons")
    plt.fill_between(df_photons.x, df_photons["16"], df_photons["84"], facecolor="blue", alpha=0.2)
    df_neurals = binned_median(features[(features.pdgId==130)& (features.eta>=eta_range[0]) & (features.eta<eta_range[1])]["puppi_weight"], features[(features.pdgId==130)& (features.eta>=eta_range[0]) & (features.eta<eta_range[1])]["graphMET_weight"], nbins=40, bins = None, x_range=(0,1))
    plt.plot(df_neurals.x, df_neurals["median"], alpha=0.7, color="green", label="neutrals")
    plt.fill_between(df_neurals.x, df_neurals["16"], df_neurals["84"], facecolor="green", alpha=0.2)
    plt.xlabel("puppi weight")
    plt.ylabel("graphMET weight")
    plt.title("eta = {} - {}".format(str(eta_range[0]),str(eta_range[1])))
    plt.legend()
    plt.savefig(plot_dir + "/" + out + "particles-puppi_vs_graphMET_weight_eta_{}_{}.png".format(str(eta_range[0]),str(eta_range[1])))
    plt.close()

    df_photons = binned_median(features[(features.pdgId==22)& (features.eta>=eta_range[0]) & (features.eta<eta_range[1])]["pT"], features[(features.pdgId==22)& (features.eta>=eta_range[0]) & (features.eta<eta_range[1])]["graphMET_weight"], nbins=50, bins = None, x_range=(0,20))
    plt.plot(df_photons.x, df_photons["median"], alpha=0.7, color="blue", label="photons")
    plt.fill_between(df_photons.x, df_photons["16"], df_photons["84"], facecolor="blue", alpha=0.2)
    df_neurals = binned_median(features[(features.pdgId==130)& (features.eta>=eta_range[0]) & (features.eta<eta_range[1])]["pT"], features[(features.pdgId==130)& (features.eta>=eta_range[0]) & (features.eta<eta_range[1])]["graphMET_weight"], nbins=50, bins = None, x_range=(0,20))
    plt.plot(df_neurals.x, df_neurals["median"], alpha=0.7, color="green", label="neutrals")
    plt.fill_between(df_neurals.x, df_neurals["16"], df_neurals["84"], facecolor="green", alpha=0.2)
    plt.xlabel("PF pT")
    plt.xscale("log")
    plt.ylabel("graphMET weight")
    plt.title("eta = {} - {}".format(str(eta_range[0]),str(eta_range[1])))
    plt.legend()
    plt.savefig(plot_dir + "/" + out + "particles-pT_vs_graphMET_weight_eta_{}_{}.png".format(str(eta_range[0]),str(eta_range[1])))
    plt.close()


    fig, ax = plt.subplots(nrows=1, ncols=1)
    xbins = np.logspace( np.log10(0.1),  np.log10(20), 40)
    ybins = np.linspace(0, 1, 40)
    counts, _, _ = np.histogram2d(features[(features.fromPV==0) & (features.charge!=0)& (features.eta>=eta_range[0]) & (features.eta<eta_range[1])]["pT"], features[(features.fromPV==0) & (features.charge!=0)& (features.eta>=eta_range[0]) & (features.eta<eta_range[1])]["graphMET_weight"], bins=(xbins, ybins))
    n_entry_pv0 = np.sum(counts)
    if n_entry_pv0 != 0:
        pcm = ax.pcolormesh(xbins, ybins, counts, norm=LogNorm())
        plt.colorbar(pcm)
    ax.set_xscale("log")
    ax.set_xlabel("pT")
    ax.set_ylabel("graphMET weight")
    ax.set_title("fromPV = 0, charged particles, eta = {} - {}".format(str(eta_range[0]),str(eta_range[1])))
    plt.savefig(plot_dir + "/" + out + "PV0-pT_vs_graphMET_weight_eta_{}_{}.png".format(str(eta_range[0]),str(eta_range[1])))
    plt.close()

    fig, ax = plt.subplots(nrows=1, ncols=1)
    xbins = np.logspace( np.log10(0.1),  np.log10(20), 40)
    ybins = np.linspace(0, 1, 40)
    counts, _, _ = np.histogram2d(features[(features.fromPV==1) & (features.charge!=0)& (features.eta>=eta_range[0]) & (features.eta<eta_range[1])]["pT"], features[(features.fromPV==1) & (features.charge!=0)& (features.eta>=eta_range[0]) & (features.eta<eta_range[1])]["graphMET_weight"], bins=(xbins, ybins))
    n_entry_pv1 = np.sum(counts)
    if n_entry_pv1 != 0:
        pcm = ax.pcolormesh(xbins, ybins, counts, norm=LogNorm())
        plt.colorbar(pcm)
    ax.set_xscale("log")
    ax.set_xlabel("pT")
    ax.set_ylabel("graphMET weight")
    ax.set_title("fromPV = 1, charged particles, eta = {} - {}".format(str(eta_range[0]),str(eta_range[1])))
    plt.savefig(plot_dir + "/" + out + "PV1-pT_vs_graphMET_weight_eta_{}_{}.png".format(str(eta_range[0]),str(eta_range[1])))
    plt.close()

    fig, ax = plt.subplots(nrows=1, ncols=1)
    xbins = np.logspace( np.log10(0.1),  np.log10(20), 40)
    ybins = np.linspace(0, 1, 40)
    counts, _, _ = np.histogram2d(features[(features.fromPV==2) & (features.charge!=0)& (features.eta>=eta_range[0]) & (features.eta<eta_range[1])]["pT"], features[(features.fromPV==2) & (features.charge!=0)& (features.eta>=eta_range[0]) & (features.eta<eta_range[1])]["graphMET_weight"], bins=(xbins, ybins))
    n_entry_pv2 = np.sum(counts)
    if n_entry_pv2 != 0:
        pcm = ax.pcolormesh(xbins, ybins, counts, norm=LogNorm())
        plt.colorbar(pcm)
    ax.set_xscale("log")
    ax.set_xlabel("pT")
    ax.set_ylabel("graphMET weight")
    ax.set_title("fromPV = 2, charged particles, eta = {} - {}".format(str(eta_range[0]),str(eta_range[1])))
    plt.savefig(plot_dir + "/" + out + "PV2-pT_vs_graphMET_weight_eta_{}_{}.png".format(str(eta_range[0]),str(eta_range[1])))
    plt.close()

    fig, ax = plt.subplots(nrows=1, ncols=1)
    xbins = np.logspace( np.log10(0.1),  np.log10(20), 40)
    ybins = np.linspace(0, 1, 40)
    counts, _, _ = np.histogram2d(features[(features.fromPV==3) & (features.charge!=0)& (features.eta>=eta_range[0]) & (features.eta<eta_range[1])]["pT"], features[(features.fromPV==3) & (features.charge!=0)& (features.eta>=eta_range[0]) & (features.eta<eta_range[1])]["graphMET_weight"], bins=(xbins, ybins))
    n_entry_pv3 = np.sum(counts)
    if n_entry_pv3 != 0:
        pcm = ax.pcolormesh(xbins, ybins, counts, norm=LogNorm())
        plt.colorbar(pcm)
    ax.set_xscale("log")
    ax.set_xlabel("pT")
    ax.set_ylabel("graphMET weight")
    ax.set_title("fromPV = 3, charged particles, eta = {} - {}".format(str(eta_range[0]),str(eta_range[1])))
    plt.savefig(plot_dir + "/" + out + "PV3-pT_vs_graphMET_weight_eta_{}_{}.png".format(str(eta_range[0]),str(eta_range[1])))
    plt.close()

    if n_entry_pv0 != 0:
        df_pv0 = binned_median(features[(features.fromPV==0) & (features.charge!=0)& (features.eta>=eta_range[0]) & (features.eta<eta_range[1])]["pT"], features[(features.fromPV==0) & (features.charge!=0)& (features.eta>=eta_range[0]) & (features.eta<eta_range[1])]["graphMET_weight"], nbins=50, bins = None, x_range=(0,20))
        plt.plot(df_pv0.x, df_pv0["median"], alpha=0.7, color="blue", label="fromPV = 0")
        plt.fill_between(df_pv0.x, df_pv0["16"], df_pv0["84"], facecolor="blue", alpha=0.2)
    if n_entry_pv1 != 0:
        df_pv1 = binned_median(features[(features.fromPV==1) & (features.charge!=0)& (features.eta>=eta_range[0]) & (features.eta<eta_range[1])]["pT"], features[(features.fromPV==1) & (features.charge!=0)& (features.eta>=eta_range[0]) & (features.eta<eta_range[1])]["graphMET_weight"], nbins=50, bins = None, x_range=(0,20))
        plt.plot(df_pv1.x, df_pv1["median"], alpha=0.7, color="green", label="fromPV = 1")
        plt.fill_between(df_pv1.x, df_pv1["16"], df_pv1["84"], facecolor="green", alpha=0.2)
    if n_entry_pv2 != 0:
        df_pv2 = binned_median(features[(features.fromPV==2) & (features.charge!=0)& (features.eta>=eta_range[0]) & (features.eta<eta_range[1])]["pT"], features[(features.fromPV==2) & (features.charge!=0)& (features.eta>=eta_range[0]) & (features.eta<eta_range[1])]["graphMET_weight"], nbins=50, bins = None, x_range=(0,20))
        plt.plot(df_pv2.x, df_pv2["median"], alpha=0.7, color="red", label="fromPV = 2")
        plt.fill_between(df_pv2.x, df_pv2["16"], df_pv2["84"], facecolor="red", alpha=0.2)
    if n_entry_pv3 != 0:
        df_pv3 = binned_median(features[(features.fromPV==3) & (features.charge!=0)& (features.eta>=eta_range[0]) & (features.eta<eta_range[1])]["pT"], features[(features.fromPV==3) & (features.charge!=0)& (features.eta>=eta_range[0]) & (features.eta<eta_range[1])]["graphMET_weight"], nbins=50, bins = None, x_range=(0,20))
        plt.plot(df_pv3.x, df_pv3["median"], alpha=0.7, color="black", label="fromPV = 3")
        plt.fill_between(df_pv3.x, df_pv3["16"], df_pv3["84"], facecolor="black", alpha=0.2)
    plt.xlabel("PF pT")
    plt.xscale("log")
    plt.ylabel("graphMET weight")
    plt.title("charged particles, eta = {} - {}".format(str(eta_range[0]),str(eta_range[1])))
    plt.legend()
    plt.savefig(plot_dir + "/" + out + "fromPV-pT_vs_graphMET_weight_eta_{}_{}.png".format(str(eta_range[0]),str(eta_range[1])))
    plt.close()

    if n_entry_pv0 != 0:
        plt.hist(features[(features.fromPV==0) & (features.charge!=0)& (features.eta>=eta_range[0]) & (features.eta<eta_range[1])]["graphMET_weight"], bins=40, range=(0,1), density=True, histtype="step", label="fromPV=0")
    if n_entry_pv1 != 0:
        plt.hist(features[(features.fromPV==1) & (features.charge!=0)& (features.eta>=eta_range[0]) & (features.eta<eta_range[1])]["graphMET_weight"], bins=40, range=(0,1), density=True, histtype="step", label="fromPV=1")
    if n_entry_pv2 != 0:
        plt.hist(features[(features.fromPV==2) & (features.charge!=0)& (features.eta>=eta_range[0]) & (features.eta<eta_range[1])]["graphMET_weight"], bins=40, range=(0,1), density=True, histtype="step", label="fromPV=2")
    if n_entry_pv3 != 0:
        plt.hist(features[(features.fromPV==3) & (features.charge!=0)& (features.eta>=eta_range[0]) & (features.eta<eta_range[1])]["graphMET_weight"], bins=40, range=(0,1), density=True, histtype="step", label="fromPV=3")
    plt.ylabel("normalized")
    plt.yscale("log")
    plt.xlabel("graphMET weight")
    plt.title("charged particles, eta = {} - {}".format(str(eta_range[0]),str(eta_range[1])))
    plt.legend()
    plt.savefig(plot_dir + "/" + out + "fromPV-graphMET_weight_eta_{}_{}.png".format(str(eta_range[0]),str(eta_range[1])))
    plt.close()

    plt.hist(features[(features.puppi_weight==0) & (features.charge!=0)& (features.eta>=eta_range[0]) & (features.eta<eta_range[1])]["graphMET_weight"], bins=40, density=True, histtype="step", label="puppi=0")
    plt.hist(features[(features.puppi_weight==1) & (features.charge!=0)& (features.eta>=eta_range[0]) & (features.eta<eta_range[1])]["graphMET_weight"], bins=40, density=True, histtype="step", label="puppi=1")
    plt.ylabel("normalized")
    plt.yscale("log")
    plt.xlabel("graphMET weight")
    plt.title("charged particles, eta = {} - {}".format(str(eta_range[0]),str(eta_range[1])))
    plt.legend()
    plt.savefig(plot_dir + "/" + out + "puppi-graphMET_weight_eta_{}_{}.png".format(str(eta_range[0]),str(eta_range[1])))
    plt.close()
