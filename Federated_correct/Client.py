from ultralytics import YOLO
import os
import torch
import argparse
import json


class Client:
    def __init__(self, node: int,config:str=None,current_global_epoch:int=0):
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
                                    epochs=local_epochs*2,
                                    batch=self.cfg["batch"],
                                    workers=self.cfg["workers"],
                                    name=f"{self.cfg['project']}/GE{self.current_global_epoch}/{self.cfg['distribution']}{node}",
                                    val=False,
                                    device=self.cfg['devices'])
        else:
            model_path = os.path.join(self.cfg["path_to_averagedmodel"], f"GE{self.current_global_epoch}_averaged.pt")
            model = YOLO(model_path)
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("local_epochs", type=int)
    parser.add_argument("node", type=int)
    parser.add_argument("current_global_epoch", type=int)
    args = parser.parse_args()
    client = Client(node=args.node,config=args.config,current_global_epoch = args.current_global_epoch)
    client.train_random_node(args.node, args.local_epochs)

