import copy
from pathlib import Path
from ultralytics import YOLO
import os
import torch
import argparse
import json
import pandas as pd


class Client:
    def __init__(self, node: int,config:str=None,current_global_epoch:int=0, model_path:str=None):
        self.node = node
        self.cfg = {}
        with open(config) as f:
            self.cfg = json.load(f)
        self.current_global_epoch = current_global_epoch

    def train_random_node(self, node:int, local_epochs: int):
        """
        Trains random Nodes for N amount of epochs and then avergaes all models and updates them.
        @param num_of_nodes:  Number of nodes we want to train each global epoch. Each node is chosen randomly
        @param local_epochs: Trains the random nodes locally for the number of epochs
        """

        if self.current_global_epoch == 0:
            model = YOLO(self.cfg["path_to_model"])
            model.train(task="detect",
                                    pretrained=False,
                                    data=f'{self.cfg["yaml_files"]}/{node}.yaml',
                                    epochs=local_epochs,
                                    batch=self.cfg["batch"],
                                    workers=self.cfg["workers"],
                                    name=f"{self.cfg['project']}/GE{self.current_global_epoch}/{self.cfg['distribution']}{node}",
                                    val=False,
                                    device=self.cfg['devices'])
        else:
            last_epoch = self.current_global_epoch-1
            model = self.average_model(last_epoch)
            model.train(task="detect",
                                    pretrained=False,
                                    data=f'{self.cfg["yaml_files"]}/{node}.yaml',
                                    epochs=local_epochs,
                                    batch=self.cfg["batch"],
                                    workers=self.cfg["workers"],
                                    name=f"{self.cfg['project']}/GE{self.current_global_epoch}/{self.cfg['distribution']}{node}",
                                    val=False,
                                    warmup_epochs=0,
                                    device=self.cfg['devices']
                                    )

        del model


    def average_model(self, last_epoch):
        nodes = self.cfg[f"current_global_epoch{last_epoch}"]
        path = Path().resolve()
        model_path = os.path.join(path,"runs/detect", self.cfg['project'], f"GE{last_epoch}")
        model_avg = YOLO(os.path.join(model_path, f"{self.cfg['distribution']}{nodes[0]}/weights/last.pt"))
        model_avg_dict = copy.deepcopy(model_avg.model.state_dict())

        train_labels = self.cfg[f"k_fold_{self.cfg['k_fold']}"]

        train_df = pd.DataFrame(train_labels).T
        categories = list(train_df.columns)
        train_df["node"] = train_df.index
        #categories.remove("node")
        train_df['node'] = train_df['node'].str.extract('(\d+)')
        train_df['node'] = pd.to_numeric(train_df['node'])  # changes the data to numbers instead of strings

        trained_df = train_df.loc[train_df["node"].isin(nodes)]

        for category in categories:
            trained_df[category] = trained_df[category] / trained_df[category].sum() / len(categories)  # calculates factr per label

        trained_df["factor"] = 0
        for category in categories:
            trained_df["factor"] += trained_df[category]  # adds up factor per label



        factor = trained_df.loc[trained_df["node"] == nodes[0], "factor"][0]
        for key in model_avg_dict.keys():
            model_avg_dict[key] = torch.mul(model_avg_dict[key], factor)

        for trained_node in nodes[1:]:
            other_model = YOLO(os.path.join(model_path, f"{self.cfg['distribution']}{trained_node}/weights/last.pt"))
            other_model = copy.deepcopy(other_model.model.state_dict())
            factor = trained_df.loc[trained_df["node"] == trained_node, "factor"][0]
            for key in model_avg_dict.keys():
                model_avg_dict[key] += torch.mul(other_model[key], factor)

        model_avg.model.load_state_dict(model_avg_dict)
        return model_avg

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("local_epochs", type=int)
    parser.add_argument("node", type=int)
    parser.add_argument("current_global_epoch", type=int)
    args = parser.parse_args()
    client = Client(node=args.node,config=args.config,current_global_epoch = args.current_global_epoch)
    client.train_random_node(args.node, args.local_epochs)

