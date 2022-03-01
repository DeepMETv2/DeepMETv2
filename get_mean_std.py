import yaml
import numpy as np 
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--config', default = None, help = "path to config")

args = parser.parse_args()
path_to_config = args.config
path_string = path_to_config.split("/")
path_string.remove(path_string[-1])

with open(path_to_config, "r") as file:
    infos = yaml.load(file, yaml.FullLoader)

train_time = infos["output"]["train_time_per_epoch"]
train_loss = infos["output"]["train_loss"]

eval_time = infos["output"]["eval_time_per_epoch"]
eval_loss = infos["output"]["eval_loss"]

train_mean = np.mean(train_time)
train_std = np.std(train_time)
best_train_loss = min(train_loss)

eval_mean = np.mean(eval_time)
eval_std = np.std(eval_time)
best_eval_loss = min(eval_loss)

if True:
        plt.plot(infos["output"]["epoch"], infos["output"]["train_loss"], label="training loss")
        plt.plot(infos["output"]["epoch"], infos["output"]["eval_loss"], label="evaluation loss")
        plt.legend()
        plt.grid()
        plt.xlabel("epochs")
        plt.ylabel("loss value")
        plt.savefig("/".join(path_string) + "/loss_new.pdf")

print("Training: \n", "best loss: ", best_train_loss, "\n", "mean time per epoch: ", train_mean, "\n", "standard deviation time per epoch: ", train_std)
print("Evaluation: \n", "best loss: ", best_eval_loss, "\n", "mean time per epoch: ", eval_mean, "\n", "standard deviation time per epoch: ", eval_std)