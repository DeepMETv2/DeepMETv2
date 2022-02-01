import yaml
import numpy as np 
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--config', default = None, help = "path to config")

parser.add_argument('--variable', default = None, help = "name of variable")

args = parser.parse_args()

with open(args.config, "r") as file:
    infos = yaml.load(file, yaml.FullLoader)

var = infos["output"][args.variable]

mean = np.mean(var)
std = np.std(var)
print("mean: ", mean, "\n", "standard deviation: ", std)