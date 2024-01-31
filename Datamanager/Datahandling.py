import copy
from glob import glob
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans
import os
import shutil
import random
import json
import argparse


class Datamanager:
    def __init__(self, path_to_scenes: str, output_path:str, filter_by:str=None):
        """
        Creates Datamanager that can create a dataset for the training of YOLOv8 Ultralytics. The data can be split
        random (IID) or dirichlet (Non-IID)
        The data has to be split by scenes and we expect that there are subfolders for each scene
        :param path_to_scenes: Path to the folders for each scene
        :param num_datasets: Num of Datasets you want to create. In Federated training this number is equal to the number of Clients you will train
        :param num_scenes_per_data: Scenes per Client
        """
        #self.cfg = {"random_state": random.randint(0, 9999)}
        self.cfg = {"random_state": 0}
        self.cfg["output_path"] = output_path
        self.cfg["filter"] = filter_by
        self.path = path_to_scenes
        self.data = {}
        self.categories = []
        self.scene_info = pd.DataFrame()
        self.datasets = []

        with open(os.path.join(self.path, "_darknet.labels")) as f:
            for x in f:
                self.categories.append(x.replace("\n", ""))


    def get_label_distribution(self, json_scenes:str) -> None:
        for scene in os.listdir(self.path):
            item_path = os.path.join(self.path, scene)
            if os.path.isdir(item_path):
                self.data[scene] = {}

        with open(f"{json_scenes}") as f:
            scene_json = pd.read_json(f)

        for scene in tqdm(self.data.keys(), desc="Getting Label Distribution"):
            for label_file in glob(os.path.join(self.path, scene, "labels","*.txt")):
                self.data[scene][label_file] = {}
                with open(label_file) as f:
                    for label in f:
                        category = self.categories[int(label[0])]
                        try:
                            self.data[scene][label_file][category] += 1
                        except:
                            self.data[scene][label_file][category] = 1


            self.data[scene] = pd.DataFrame.from_dict(self.data[scene]).T
            self.data[scene] = self.data[scene].fillna(0)
            self.data[scene]["name"] = pd.Series((list(self.data[scene].index))).values
            self.data[scene]["scene"] = np.nan
            self.data[scene] = self.data[scene].fillna(f"{scene}")
            description = scene_json.loc[scene_json["token"] == scene]["description"].values[0]
            self.data[scene]["description"] = np.nan
            self.data[scene]["description"] = self.data[scene]["description"].fillna(f"{description}")

        self.data = pd.concat(self.data)
        self.data = self.data.fillna(0)

        scenes = self.data["scene"].drop_duplicates()

        for scene in tqdm(scenes, desc="Calculating the sum for all categories"):
            self.scene_info.loc[scene, "name"] = scene
            self.scene_info.loc[scene, "scene"] = scene
            description = scene_json.loc[scene_json["token"] == scene]["description"].values[0]
            self.scene_info.loc[scene, "description"] = description
            for category in self.categories:
                self.scene_info.loc[scene, category] = self.data[category].loc[self.data["scene"] == scene].sum()

        self.data = pd.concat([self.data, self.scene_info])
        self.data["weather"] = np.nan
        self.data.loc[self.data["description"].str.contains("rain", case=False) & ~self.data["description"].str.contains("night", case=False), "weather"] = "rain"
        self.data.loc[self.data["description"].str.contains("night", case=False) & ~self.data["description"].str.contains("rain", case=False), "weather"] = "night"
        self.data.loc[self.data["description"].str.contains("night", case=False) & self.data["description"].str.contains("rain", case=False), "weather"] = "rainynight"
        self.data.loc[~self.data["description"].str.contains("night", case=False) & ~self.data["description"].str.contains("rain", case=False), "weather"] = "day"
        self.scene_info["weather"] = np.nan
        self.scene_info.loc[self.scene_info["description"].str.contains("rain", case=False) & ~self.scene_info["description"].str.contains("night",case=False), "weather"] = "rain"
        self.scene_info.loc[self.scene_info["description"].str.contains("night", case=False) & ~self.scene_info["description"].str.contains("rain",case=False), "weather"] = "night"
        self.scene_info.loc[self.scene_info["description"].str.contains("night", case=False) & self.scene_info["description"].str.contains("rain",case=False), "weather"] = "rainynight"
        self.scene_info.loc[~self.scene_info["description"].str.contains("night", case=False) & ~self.scene_info["description"].str.contains("rain",case=False), "weather"] = "day"

        if self.cfg["filter"] != None:
            self.data = self.data.loc[self.data["weather"] == self.cfg["filter"]]
            self.scene_info = self.scene_info.loc[self.scene_info["weather"] == self.cfg["filter"]]


    def create_datasets(self, num_of_scenes_per_dataset: int = 10, num_of_datasets: int = 10, distribution: str = "even") -> None:
        """
        Creates dataset in the data class. Can be used to create different dataset distribution according to the
        distribution Parameter
        @param num_of_scenes_per_dataset:
        @param num_of_datasets:
        @param distribution:
        @return: nothing
        """
        self.datasets = []
        self.cfg["num_of_scenes_per_dataset"] = num_of_scenes_per_dataset
        self.cfg["num_of_datasets"] = num_of_datasets
        self.cfg["distribution"] = distribution

        if len(self.scene_info) < num_of_scenes_per_dataset*num_of_datasets:
            raise Exception("Not enough scenes available to create data for each node")


        if len(self.scene_info) < num_of_scenes_per_dataset*num_of_datasets:
            raise Exception("Not enough scenes available to create data for each node")

        self.datasets =[]

        if distribution == "random":
            sub_df = self.scene_info.sample(n=num_of_scenes_per_dataset * num_of_datasets, random_state=self.cfg["random_state"])
            for i in range(num_of_datasets):
                temp_df = sub_df.sample(n=num_of_scenes_per_dataset, random_state=self.cfg["random_state"])
                sub_df = sub_df.drop(temp_df.index)
                self.datasets.append(temp_df)


        if distribution == "dirichlet":
            if int(num_of_datasets/len(self.categories)) < 1:
                raise Exception("For dirichlet distribution the number of datasets has to be bigger than the number of Categories!")
            sorted_data = copy.deepcopy(self.scene_info)
            for i in range(num_of_datasets):
                self.datasets.append([])
            for i in range(num_of_scenes_per_dataset):
                for j in range(num_of_datasets):
                    category = self.categories[j % len(self.categories)]
                    highest_value = sorted_data.sort_values(by=f"{category}", ascending=False).head(1)
                    sorted_data = sorted_data.drop(highest_value.index)
                    highest_value["class"] = category
                    self.datasets[j].append(highest_value)
            for i in range(num_of_datasets):
                self.datasets[i] = pd.concat(self.datasets[i])



    def copy_files_fed_and_cent(self,path_to_dataset:str,k:int=5,copy_files:bool=False):
        self.cfg["output_path"] = os.path.join(self.cfg["output_path"], self.cfg["distribution"])
        self.cfg["k_fold"] = k
        if os.path.isdir(os.path.join(self.cfg["output_path"])):
            raise Exception("Directory already exist use a different output_path for this distribution")

        self.cfg["path_to_dataset"] = path_to_dataset

        files = pd.DataFrame(glob(os.path.join(path_to_dataset, "**", "*")))
        files["filename"] = files[0].apply(os.path.basename)
        files["filename"] = files["filename"].apply(lambda x: x[:-4])

        data_without_scenes = self.data.loc[~(self.data["scene"] == self.data["name"])]

        for idx, dataset in tqdm(enumerate(self.datasets), desc=f"Copying data for Nodes:"):
            if copy_files:
                for k_fold in range(k):
                    os.makedirs(os.path.join(self.cfg["output_path"],f"k_fold_{k_fold}", str(idx), "images", "train"))
                    os.makedirs(os.path.join(self.cfg["output_path"],f"k_fold_{k_fold}", str(idx), "images", "val"))
                    os.makedirs(os.path.join(self.cfg["output_path"],f"k_fold_{k_fold}", str(idx), "labels", "train"))
                    os.makedirs(os.path.join(self.cfg["output_path"],f"k_fold_{k_fold}", str(idx), "labels", "val"))
            if self.cfg["distribution"] == "dirichlet":
                category = dataset["class"].drop_duplicates()[0]
            dataset = copy.deepcopy(data_without_scenes.loc[data_without_scenes["scene"].isin(dataset["scene"])])
            dataset["filename"] = dataset["name"].apply(os.path.basename)
            dataset["filename"] = dataset["filename"].apply(lambda x: x[:-4])
            dataset["output_path"] = self.cfg["output_path"]
            dataset["distribution"] = self.cfg["distribution"]
            dataset["node"] = str(idx)
            dataset["sum_classes"] = dataset[self.categories].apply(func=sum, axis=1)
            if "dirichlet" in self.cfg["distribution"]:
                dataset = dataset[dataset[self.categories].apply(lambda x: (x > 0).any(), axis=1)] #take all labelled files
                if "ascending" not in locals():
                    ascending = True
                if "dirichlet" in self.cfg["distribution"]:
                    if "ascending_diri" not in locals():
                        ascending_diri = [True, True]
                if "dirichlet" in self.cfg["distribution"]:
                    ascender = (idx % len(self.categories))
                    dataset.sort_values(by=self.categories[ascender], ascending=ascending_diri[ascender], inplace=True)
                    #dataset = dataset.iloc[:400]
                    ascending_diri[ascender] = not ascending_diri[ascender]
                else:
                    dataset.sort_values(by="sum_classes", ascending=ascending, inplace=True)
                    ascending = not ascending
                    dataset = dataset.iloc[:int(len(dataset)*0.4)]
            else:
                dataset = dataset[dataset[self.categories].apply(lambda x: (x > 0).all(), axis=1)] #only use labeled images with both labels
            if "dirichlet" == self.cfg["distribution"]:
                dataset["class"] = category
                other_categories = [x for x in self.categories if x != category]
                dataset = dataset[dataset.apply(lambda x: (x[category] > x[other_categories]).all(), axis=1)] #delete images with more labels for the non dirichlet class
                dataset = dataset.iloc[:400]
            #dataset_train = dataset.sample(frac=0.8, random_state=self.cfg["random_state"])

            dataset = dataset.sample(frac=1, random_state=self.cfg["random_state"]) #shuffles dataset
            cutoff = 400  #can be set to a different value
            dataset = dataset.iloc[:cutoff]
            for k_fold in range(k):
                dataset_val = dataset.iloc[int(k_fold*(cutoff/k)):int((k_fold+1)*(cutoff/k))] #kfold for val set
                dataset_train = dataset.drop(dataset_val.index) #get train dataset
                dataset_train["path_to_image"] = files.loc[files["filename"].isin(dataset_train["filename"])][0].values  #gets all images
                dataset_val["path_to_image"] = files.loc[files["filename"].isin(dataset_val["filename"])][0].values  #gets all images
                dataset_train["train"] = "train"
                dataset_val["train"] = "val"
                dataset = pd.concat([dataset_train, dataset_val])
                dataset["node"] = f"{idx}"
                dataset["k_fold"] = k_fold
                self.datasets[idx] = dataset

                info_datasets = {}
                info_datasets[f"k_fold_{k_fold}"] = {}
                info_datasets[f"k_fold_{k_fold}"][f"node{idx}"] = dataset.loc[dataset["train"] == "train"][self.categories].sum().to_dict()
                try:
                    self.cfg[f"k_fold_{k_fold}"].update(info_datasets[f"k_fold_{k_fold}"])
                except:
                    self.cfg.update(info_datasets)


                if copy_files:
                    self.datasets[idx].apply(func=copy_file, axis=1)


    def plot_dataset(self):
        os.makedirs(os.path.join(self.cfg["output_path"], "Plots"))
        output_path = os.path.join(self.cfg["output_path"], "Plots")
        for idx, dataset in tqdm(enumerate(self.datasets), desc=f"Creating Plots/Config for each Node:"):
            cfg_dataset = {}
            cfg_dataset[f"Node_{idx}"] = {}
            cfg_dataset[f"Node_{idx}"]["train"] = {}
            cfg_dataset[f"Node_{idx}"]["val"] = {}
            dataset_plot = dataset.loc[dataset["train"] == "train"][self.categories].sum()
            fig = dataset_plot.plot.bar(rot=0).get_figure()
            fig.savefig(os.path.join(output_path, f"train{idx}.png"))
            dataset_plot = dataset.loc[dataset["train"] == "val"][self.categories].sum()
            cfg_dataset[f"Node_{idx}"]["val"]["sum"] = dataset_plot.to_dict()
            fig = dataset_plot.plot.bar(rot=0).get_figure()
            fig.savefig(os.path.join(output_path, f"val{idx}.png"))
        if self.cfg.get("Datasets") == None:
            self.cfg["Datasets"] = {}
        for idx, dataset in enumerate(self.datasets):
            cfg_dataset = {}
            cfg_dataset[f"Node_{idx}"] = {}
            cfg_dataset[f"Node_{idx}"]["scenes"] = list(dataset.loc[~dataset["node"].str.contains("shared")]["scene"].drop_duplicates().values)#gets all original scenes
            cfg_dataset[f"Node_{idx}"]["train"] = {}
            cfg_dataset[f"Node_{idx}"]["val"] = {}

            dataset_dict = dataset.loc[dataset["train"] == "train"][self.categories]
            cfg_dataset[f"Node_{idx}"]["train"]["num_samples"] = len(dataset_dict)
            cfg_dataset[f"Node_{idx}"]["train"]["sum"] = dataset_dict.sum().to_dict()
            cfg_dataset[f"Node_{idx}"]["train"]["std"] = dataset_dict.std().to_dict()
            cfg_dataset[f"Node_{idx}"]["train"]["mean"] = dataset_dict.mean().to_dict()
            cfg_dataset[f"Node_{idx}"]["train"]["median"] = dataset_dict.median().to_dict()

            dataset_dict = dataset.loc[dataset["train"] == "val"][self.categories]
            cfg_dataset[f"Node_{idx}"]["val"]["num_samples"] = len(dataset_dict)
            cfg_dataset[f"Node_{idx}"]["val"]["sum"] = dataset_dict.sum().to_dict()
            cfg_dataset[f"Node_{idx}"]["val"]["std"] = dataset_dict.std().to_dict()
            cfg_dataset[f"Node_{idx}"]["val"]["mean"] = dataset_dict.mean().to_dict()
            cfg_dataset[f"Node_{idx}"]["val"]["median"] = dataset_dict.median().to_dict()
            if self.cfg['one_sided']:
                category = dataset["class"].drop_duplicates()[0]
                category = self.categories[category]
                for cat in self.categories:
                    if cat != category:
                        cfg_dataset[f"Node_{idx}"]["train"]["sum"][category] = 0
                        cfg_dataset[f"Node_{idx}"]["train"]["std"][category] = 0
                        cfg_dataset[f"Node_{idx}"]["train"]["mean"][category] = 0
                        cfg_dataset[f"Node_{idx}"]["train"]["median"][category] = 0
            self.cfg["Datasets"].update(cfg_dataset)

        with open(os.path.join(self.cfg["output_path"], "config.json"), "w") as outfile:
            json.dump(self.cfg, outfile, indent=4)





    def create_yaml(self, path_to_yaml:str):
        """
        Creates all required YAML files for each k-fold. Is required to be used later in the training.
        Parameters
        ----------
        path_to_yaml :Path to folder for all YAML files, which are required for training

        Returns None
        -------
        """
        self.cfg["yaml_files"] = os.path.join(path_to_yaml, self.cfg["distribution"])
        if os.path.isdir(self.cfg["yaml_files"]):
            raise Exception("path to yaml already exists check if the Yaml are needed first")
        for k in range(5):
            os.makedirs(f"{self.cfg['yaml_files']}/k_fold_{k}")
            for idx, dataset in enumerate(self.datasets):
                with open(os.path.join(self.cfg["yaml_files"],f"k_fold_{k}", f"{idx}.yaml"), "w") as f:
                    f.write(f"path: {self.cfg['output_path']}\n")
                    f.write(f"train: k_fold_{k}/{idx}/images/train\n")
                    f.write(f"val: k_fold_{k}/{idx}/images/val\n")
                    f.write("names:\n")
                    f.write("  0: human.pedestrian\n")
                    f.write("  1: vehicle.car\n")

        for k in range(5):
            with open(os.path.join(self.cfg["yaml_files"],f"k_fold_{k}", f"all.yaml"), "w") as f: # writes the all.yaml file
                f.write(f"path: {self.cfg['output_path']}/k_fold_{k}\n")
                f.write(f"train: [0/images/train")
                for idx in range(1,self.cfg['num_of_datasets']):
                    f.write(f",{idx}/images/train")
                f.write(f"]\n")
                f.write(f"val: [0/images/val")
                for idx in range(1, self.cfg['num_of_datasets']):
                    f.write(f",{idx}/images/val")
                f.write(f"]\n")
                f.write("names:\n")
                f.write("  0: human.pedestrian\n")
                f.write("  1: vehicle.car\n")
        with open(os.path.join(self.cfg["output_path"], "config.json"), "w") as outfile:
            json.dump(self.cfg, outfile, indent=4)






def copy_file(row):
    output_path = os.path.join(row["output_path"],f"k_fold_{row['k_fold']}", str(row["node"]))
    shutil.copy2(row["name"], os.path.join(output_path, "labels", row["train"]))
    shutil.copy2(row["path_to_image"], os.path.join(output_path, "images", row["train"]))


if __name__ == '__main__':
    path_to_scenes = "" #Path to the folders for all scenes in Darknetformat
    output_path = "" #Output Path for the created subfolders for all Clients
    distribution = "" #distribution you want to use current available are [random(IID), dirichlet (Non-IID)] check out the Datamanager for more distributions!
    path_to_json = "" #file to all scenes
    path_to_dataset = "" #path where all images are stored. For Nuscenes this is the samples folder!
    data = Datamanager(path_to_scenes=path_to_scenes,
                       output_path=output_path)
    data.get_label_distribution(json_scenes=path_to_json)
    data.create_datasets(num_of_scenes_per_dataset=10, num_of_datasets=10, distribution="random")
    data.copy_files_fed_and_cent(path_to_dataset=path_to_dataset,copy_files=True)
    data.create_yaml(path_to_yaml="yaml_files") #creates the required yaml files at the path_to_yaml folder
