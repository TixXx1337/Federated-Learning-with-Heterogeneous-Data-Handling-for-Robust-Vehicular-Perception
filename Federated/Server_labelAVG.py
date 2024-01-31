from glob import glob
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
from pathlib import Path

path_to_compiler = "" #is required since the Training otherwise runs out of memory, since YOlov8 doesnt use a correct garbage collector
path_to_client_File = "" #Path to Client File e.g. [path/Client.py]

class Server:
    def __init__(self, nodes: int,
                 path_to_config: str,
                 path_to_averagedmodel: str,
                 devices="cpu", batch: int = 16,
                 workers: int = 8,
                 path_to_model: str = "yolov8n.yaml",
                 path_to_yaml: str = None,
                 current_global_epoch: int = 0
                 ):
        """
        @param nodes: Number of Nodes you want for the training.
        @param path_to_config: Config created by Datamanager. IMPORTANT Sets the path for training and averaging and is necessary fot creating yaml files
        @param path_to_averagedmodel:
        @param devices: Device(s) you want to train on can be "cpu" or the CUDA Devices as list [0,1,2]
        @param batch: Number of Batches pert training. Use -1 for Auto Batch
        @param workers: Number of Workers equal the Number of Dataloaders. If Multiple GPUS are use each GPU gets N Workers.
        @param path_to_model: Path to the model you want to train. Please work with an absolute path here
        @param current_global_epoch: Current Global Epoch. Can be used to restart from specific Global Epoch if the training was stopped.
        """
        with open(path_to_config) as f:
            self.cfg = json.load(f)
        if self.cfg.get("yaml_files") == None:
            self.cfg["yaml_files"] = path_to_yaml
        self.cfg["batch"] = batch
        self.cfg["workers"] = workers
        self.cfg["path_to_model"] = path_to_model
        self.cfg["path_to_averagedmodel"] = os.path.join(path_to_averagedmodel, "label_averaged", f'k_fold_{self.cfg["k_fold"]}', self.cfg["distribution"])
        self.cfg["project"] = f'{self.cfg["distribution"]}/k_fold_{self.cfg["k_fold"]}'
        self.cfg["devices"] = devices
        self.cfg["averaged_model"] = {}
        self.nodes = nodes
        self.current_global_epoch = current_global_epoch
        # os.makedirs(os.path.join(path_to_averagedmodel, "config"))
        if self.current_global_epoch == 0:
            os.makedirs(f"{self.cfg['path_to_averagedmodel']}")
        self.config_for_training = os.path.join(self.cfg["path_to_averagedmodel"],"config.json")
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



    def create_plot(self):
        self.metrices = {}
        model = YOLO(self.cfg["path_to_model"])
        self.metrices[0] = model.val(data=shared_yaml)
        for file in glob(f"{self.cfg['path_to_averagedmodel']}/*.pt"):
            model = YOLO(file)
            epoch = re.findall(r'\d+', file)
            self.metrices[int(epoch[1])] = model.val(data=shared_yaml, device=0)

        self.sorted_metrices = {}
        for key in sorted(self.metrices.keys()):
            self.sorted_metrices[key] = self.metrices[key]

        self.results = pd.DataFrame()
        for key in self.sorted_metrices.keys():
            self.results.loc[key, "mAP50"] = self.sorted_metrices[key].box.map50
            self.results.loc[key, "mAP50-95"] = self.sorted_metrices[key].box.map
            self.results.loc[key, "epoch"] = key

        plt.style.use('fivethirtyeight')
        plt.figure(figsize=(8, 6))
        plt.plot(self.results['epoch'], self.results['mAP50'], label='mAP50')
        plt.plot(self.results['epoch'], self.results['mAP50-95'], label='mAP50-95')
        plt.xlabel('Global Epoch')
        plt.ylabel('mAP')
        title = f"Datasharing for {self.cfg['distribution']} with shared {self.cfg['shared']} with {self.cfg['num_of_clusters']} clusters"
        plt.title(f"{title}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{self.cfg['path_to_averagedmodel']}/results.png")
        self.results.to_csv(f"{self.cfg['path_to_averagedmodel']}/results.csv")
        plt.show()

    def create_random_nodes(self, num_of_nodes):
        for i in range(self.current_global_epoch, self.current_global_epoch+20):
            self.cfg[f"current_global_epoch{i}"] = random.sample(range(0, self.nodes), num_of_nodes)
        with open(self.config_for_training, "w") as f:
            json.dump(self.cfg, f, indent=4)






