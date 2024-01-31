import glob
from pathlib import Path
import copy
import json
import numpy as np
import random
import os
from ultralytics import YOLO
import torch
import subprocess
import pickle
import re
from matplotlib import pyplot as plt
import pandas as pd

path_to_compiler = "" #is required since the Training otherwise runs out of memory, since YOlov8 doesnt use a correct garbage collector
path_to_client_File = "" #Path to Client File e.g. [path/Client.py, path/Client_FedProx.py]

class Server:
    def __init__(self, nodes: int,
                 path_to_config: str,
                 path_to_averagedmodel: str,
                 devices="cpu", batch: int = 16,
                 workers: int = 8, project: str = None,
                 path_to_model: str = "yolov8n.yaml",
                 path_to_yaml: str = None,
                 current_global_epoch: int = 0):
        """
        Server class which calls the the independent Clients
        @param nodes: Number of Nodes you want for the training.
        @param path_to_config: Config created by Datamanager. IMPORTANT Sets the path for training and averaging and is necessary fot creating yaml files
        @param path_to_averagedmodel:
        @param devices: Device(s) you want to train on can be "cpu" or the CUDA Devices as list [0,1,2]
        @param batch: Number of Batches pert training. Use -1 for Auto Batch
        @param workers: Number of Workers equal the Number of Dataloaders. If Multiple GPUS are use each GPU gets N Workers.
        @param project: Project Name you want to use for the run.
        @param path_to_model: Path to the model you want to train. Please work with an absolute path here
        @param current_global_epoch: Current Global Epoch. Can be used to restart from specific Global Epoch if the training was stopped.
        """
        with open(path_to_config) as f:
            self.cfg = json.load(f)
        self.cfg["batch"] = batch
        if path_to_yaml != None:
            self.cfg["yaml_files"] = path_to_yaml  # not needed if create yaml files is called!
        self.cfg["workers"] = workers
        self.cfg["project"] = f'{self.cfg["distribution"]}/k_fold_{self.cfg["k_fold"]}'
        self.cfg["path_to_model"] = path_to_model
        self.cfg["path_to_averagedmodel"] = os.path.join(path_to_averagedmodel,self.cfg["distribution"],f"k_fold_{self.cfg['k_fold']}")
        self.cfg["devices"] = devices
        self.nodes = nodes
        self.current_global_epoch = current_global_epoch
        # os.makedirs(os.path.join(path_to_averagedmodel, "config"))
        if os.path.isdir(f"{self.cfg['path_to_averagedmodel']}") and self.current_global_epoch == 0:
            raise Exception("averaged model path already exists. Make Sure you dont overwrite existing results")
        if self.current_global_epoch == 0:
            os.makedirs(f"{self.cfg['path_to_averagedmodel']}")
        self.config_for_training = os.path.join(self.cfg["path_to_averagedmodel"], "config.json")
        with open(self.config_for_training, "w") as f:
            json.dump(self.cfg, f, indent=4)



    def train_random_nodes(self, num_of_nodes, local_epochs: int):
        try:
            random_nodes = self.cfg[f"current_global_epoch{self.current_global_epoch}"]
        except:
            self.create_random_nodes(num_of_nodes)
            random_nodes = self.cfg[f"current_global_epoch{self.current_global_epoch}"]
        for node in random_nodes:
            subprocess.run(
                [f"{path_to_compiler}",
                 f"{path_to_client_File}",
                 self.config_for_training, f"{local_epochs}", f"{node}", f"{self.current_global_epoch}"])


    def create_random_nodes(self, num_of_nodes):
        for i in range(self.current_global_epoch, self.current_global_epoch + 20):
            self.cfg[f"current_global_epoch{i}"] = random.sample(range(0, self.nodes), num_of_nodes)
        with open(self.config_for_training, "w") as f:
            json.dump(self.cfg, f, indent=4)


