import json
import os.path as osp
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_cluster import radius_graph, knn_graph
import torch_geometric.transforms as T
from tqdm import tqdm
import argparse
import utils
import model.net as net

import model.data_loader as data_loader
from model.nano_loader import METDataset
from evaluate import evaluate

# import warnings
import time

# warnings.simplefilter("ignore")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--restore_file",
    default=None,
    help="Optional, to reload a model before training, otptions are 'best' and 'last'",
)
parser.add_argument(
    "--config",
    default="config.yaml",
    help="Name of a config in the configs/ directory, this config is used for the model and training",
)


def train(model, device, optimizer, scheduler, loss_fn, dataloader, epoch, net_info):
    model.train()
    loss_avg_arr = []
    loss_avg = utils.RunningAverage()
    start_time = time.time()

    with tqdm(total=len(dataloader)) as t:
        for data in dataloader:
            optimizer.zero_grad()
            data = data.to(device)

            if net_info["graph"]["static"]:
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

            elif net_info["graph"]["dynamic"]:
                edge_index = None
            else:
                print("Graph not defined.")

            result = model(data.x, edge_index=edge_index, batch=data.batch)

            loss = loss_fn(result, data.x, data.y, data.batch, net_info["loss"])
            loss.backward()
            optimizer.step()
            # update the average loss
            loss_avg_arr.append(loss.cpu().item())
            loss_avg.update(loss.cpu().item())
            t.set_postfix(loss="{:05.3f}".format(loss_avg()))
            t.update()
            # torch.cuda.empty_cache()

    training_time = time.time() - start_time
    mean_loss = np.mean(loss_avg_arr)
    scheduler.step(mean_loss)
    print("Training epoch: {:02d}, MSE: {:.4f}".format(epoch, mean_loss))
    return mean_loss, training_time


if __name__ == "__main__":
    args = parser.parse_args()

    info_handler = utils.info_handler(args.config)
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

    train_dl = dataloaders["train"]
    test_dl = dataloaders["test"]
    print("Number of training batches:", len(train_dl))
    print("Number of test batches:", len(test_dl))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Training performed on:", device)
    # restriction on gpu memory usage
    torch.cuda.set_per_process_memory_fraction(0.5)

    print("Using", net_info["graph"]["layer"], "layer.")
    model = net.Net(net_info).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_info["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=train_info["factor"],
        patience=train_info["patience"],
        threshold=train_info["threshold"],
    )
    first_epoch = 0
    best_validation_loss = 10e7

    loss_fn = net.loss_fn
    metrics = net.metrics

    model_dir = osp.join(os.environ["PWD"], data_info["output_dir"])

    # reload weights from restore_file if specified
    if args.restore_file is not None:
        ckpt_file = osp.join(model_dir, args.restore_file + ".pth.tar")
        ckpt = utils.load_checkpoint(ckpt_file, model, optimizer, scheduler)
        first_epoch = ckpt["epoch"]
        print("Restarting training from epoch", first_epoch)
        with open(osp.join(model_dir, "metrics_val_best.json")) as restore_metrics:
            best_validation_loss = json.load(restore_metrics)["loss"]

    for epoch in range(first_epoch + 1, train_info["max_epoch"] + 1):
        print("Current best loss:", best_validation_loss)
        if "_last_lr" in scheduler.state_dict():
            print("Learning rate:", scheduler.state_dict()["_last_lr"][0])

        # compute one epoch: one full pass over the training set
        training_loss, training_time = train(
            model, device, optimizer, scheduler, loss_fn, train_dl, epoch, net_info
        )

        # save model weights
        utils.save_checkpoint(
            {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optim_dict": optimizer.state_dict(),
                "sched_dict": scheduler.state_dict(),
            },
            is_best=False,
            checkpoint=model_dir,
        )

        # evaluate for one epoch on validation set
        test_metrics, resolutions, evaluation_time = evaluate(
            model, device, loss_fn, test_dl, metrics, net_info, model_dir, out=""
        )

        validation_loss = test_metrics["loss"]

        is_best = validation_loss <= best_validation_loss

        # if new best validation loss, save as best model weights
        if is_best:
            print("Found new best loss!")
            best_validation_loss = validation_loss

            # Save weights
            utils.save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "optim_dict": optimizer.state_dict(),
                    "sched_dict": scheduler.state_dict(),
                },
                is_best=True,
                checkpoint=model_dir,
            )

            # save jitted model for use with sonic
            model_scripted = torch.jit.script(model)
            model_scripted.save(osp.join(model_dir, "graphMET.pt"))

            # save best validation metrics in a json file in the model directory
            utils.save_dict_to_json(
                test_metrics, osp.join(model_dir, "metrics_val_best.json")
            )
            utils.save(resolutions, osp.join(model_dir, "best.resolutions"))

        utils.save_dict_to_json(
            test_metrics, osp.join(model_dir, "metrics_val_last.json")
        )
        utils.save(resolutions, osp.join(model_dir, "last.resolutions"))

        # save loss to yaml output file
        info_handler.add_epoch(
            epoch,
            float(training_loss),
            float(validation_loss),
            float(training_time),
            float(evaluation_time),
        )
        info_handler.save_infos(str(model_dir))
        # generate loss plot if more then one training epoch
        if epoch != 1:
            info_handler.plot_loss(str(model_dir))
