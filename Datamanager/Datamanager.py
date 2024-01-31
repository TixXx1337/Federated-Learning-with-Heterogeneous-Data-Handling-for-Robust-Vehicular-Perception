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
        random, even, dirichlet or clustered.
        The data has to be split by scenes and we expect that there are subfolders for each scene in the
        :param path_to_scenes:
        :param num_datasets:
        :param num_scenes_per_data:
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


    def get_label_distribution(self, path_to_scenes:str) -> None:
        for scene in os.listdir(self.path):
            item_path = os.path.join(self.path, scene)
            if os.path.isdir(item_path):
                self.data[scene] = {}

        with open(f"{path_to_scenes}") as f:
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
                        # try:
                        #    labels[category] += 1
                        # except:
                        #    labels[category] = 1
            # self.data[scene]["categories"] = labels

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
        if distribution == "cluster":
            num_clusters = len(self.categories)
            kmeans = KMeans(n_clusters=num_clusters, random_state=self.cfg["random_state"])
            kmeans.fit(self.scene_info.drop(["scene", "name", "movable.object"], axis=1))
            split_dataframes = []
            labels = kmeans.labels_
            self.scene_info['Cluster'] = labels
            for cluster_label in range(num_clusters):
                cluster_df = self.scene_info[self.scene_info['Cluster'] == cluster_label].copy()
                cluster_df.drop('Cluster', axis=1, inplace=True)
                split_dataframes.append(cluster_df)

            for cluster in split_dataframes:
                if len(cluster) < num_of_scenes_per_dataset:
                    raise Exception("Clustered is not possible as each label is not evenly distributed")

            for cluster in split_dataframes:
                self.datasets.append(cluster.sample(n=num_of_scenes_per_dataset))

        if distribution == "kmeans_even":
            if (num_of_scenes_per_dataset % len(self.categories)) != 0:
                raise Exception(
                    "For even distribution the number of Scenes has to be a multiplier of Category numbers!")
            num_clusters = len(self.categories)
            kmeans = KMeans(n_clusters=num_clusters, random_state=self.cfg["random_state"])
            kmeans.fit(self.scene_info.drop(["scene", "name"], axis=1))
            split_dataframes = []
            labels = kmeans.labels_
            self.scene_info['Cluster'] = labels
            for cluster_label in range(num_clusters):
                cluster_df = self.scene_info[self.scene_info['Cluster'] == cluster_label].copy()
                cluster_df.drop('Cluster', axis=1, inplace=True)
                split_dataframes.append(cluster_df)
            for cluster in split_dataframes:
                if len(cluster) < int(num_of_scenes_per_dataset / num_of_datasets):
                    raise Exception("Clustered is not possible as some clusters are too small\n"
                                    "Try using random or clustered data distribution")

            for i in range(num_of_datasets):
                data_list = []
                for idx, cluster in enumerate(split_dataframes):
                    subset = cluster.sample(n=int(num_of_scenes_per_dataset / num_clusters), random_state=self.cfg["random_state"])
                    split_dataframes[idx] = split_dataframes[idx].drop(subset.index)
                    data_list.append(subset)
                self.datasets.append(pd.concat(data_list))

        if distribution == "random":
            sub_df = self.scene_info.sample(n=num_of_scenes_per_dataset * num_of_datasets, random_state=self.cfg["random_state"])
            for i in range(num_of_datasets):
                temp_df = sub_df.sample(n=num_of_scenes_per_dataset, random_state=self.cfg["random_state"])
                sub_df = sub_df.drop(temp_df.index)
                self.datasets.append(temp_df)

        if distribution == "heterogenous":
            sub_df = self.scene_info.sample(n=num_of_scenes_per_dataset * num_of_datasets, random_state=self.cfg["random_state"])


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

        if distribution == "scenes":
            weather_cond = ["rain","night","day"]
            datasets_weather = {}
            for weather in weather_cond:
                datasets_weather[weather] = self.scene_info.loc[self.scene_info["weather"] == f"{weather}"]
            self.datasets = {}
            for weather in weather_cond:
                self.datasets[weather] = []
                if int(len(datasets_weather[weather]) / num_of_scenes_per_dataset) < num_of_datasets:
                    print(f"Cant create {num_of_datasets} for weather {weather} use create {int(len(datasets_weather[weather]) / num_of_scenes_per_dataset)} instead")
                    fixed_num_of_datasets = int(len(datasets_weather[weather]) / num_of_scenes_per_dataset)
                else:
                    fixed_num_of_datasets = num_of_datasets
                sub_df = datasets_weather[weather]
                for i in range(fixed_num_of_datasets):
                    temp_df = sub_df.sample(n=num_of_scenes_per_dataset, random_state=self.cfg["random_state"])
                    sub_df = sub_df.drop(temp_df.index)
                    self.datasets[weather].append(temp_df)

    def label_aware_sharing(self, path_to_dataset: str,datasets_seen: int = 8, shared_data: float = 0,
                            num_of_clusters:int =3,mix_cluster:bool=None, meet_other_nodes:bool=False) -> None:

        self.cfg["distribution"] = f"{self.cfg['distribution']}_intelligentshared"
        self.cfg["output_path"] = os.path.join(self.cfg["output_path"], "ensemble_datasharing", self.cfg["distribution"], f"shared{shared_data}")
        if os.path.isdir(self.cfg["output_path"]):
            raise Exception("Directory already exist use a different output_path for this distribution")
        self.cfg["shared"] = shared_data
        self.cfg["Datasharing"] = True
        if shared_data <= 0 or shared_data > 1:
            raise Exception("Shared amount too big or not set!")

        files = pd.DataFrame(glob(os.path.join(path_to_dataset, "**", "*")))
        files["filename"] = files[0].apply(os.path.basename)
        files["filename"] = files["filename"].apply(lambda x: x[:-4])

        data_without_scenes = self.data.loc[~(self.data["scene"] == self.data["name"])]
        for idx, dataset in tqdm(enumerate(self.datasets), desc=f"Copying data for Nodes:"):
            os.makedirs(os.path.join(self.cfg["output_path"], str(idx), "images", "train"))
            os.makedirs(os.path.join(self.cfg["output_path"], str(idx), "images", "val"))
            os.makedirs(os.path.join(self.cfg["output_path"], str(idx), "labels", "train"))
            os.makedirs(os.path.join(self.cfg["output_path"], str(idx), "labels", "val"))
            category = dataset["class"].drop_duplicates()[0]
            dataset = data_without_scenes.loc[data_without_scenes["scene"].isin(dataset["scene"])]
            dataset["filename"] = dataset["name"].apply(os.path.basename)
            dataset["filename"] = dataset["filename"].apply(lambda x: x[:-4])
            dataset["output_path"] = self.cfg["output_path"]
            dataset["distribution"] = self.cfg["distribution"]
            dataset["class"] = category
            dataset["node"] = str(idx)
            dataset_train = dataset.sample(frac=0.8, random_state=self.cfg["random_state"])
            dataset_val = dataset.drop(dataset_train.index)
            #dataset_train["path_to_image"] = files.loc[files["filename"].isin(dataset_train["filename"])][0].values  # gets all images
            #dataset_val["path_to_image"] = files.loc[files["filename"].isin(dataset_val["filename"])][0].values  # gets all images
            dataset_train["train"] = "train"
            dataset_val["train"] = "val"
            dataset = pd.concat([dataset_train, dataset_val])
            self.datasets[idx] = dataset

        self.cfg["clusters_for_nodes"] = {}
        for idx, dataset in enumerate(self.datasets):
            cluster = idx % num_of_clusters

            if "dirichlet" in self.cfg["distribution"]:
                if mix_cluster == None:
                    raise Exception("Paramater mix_clusters is required for dirichlet distribution!")
                elif mix_cluster:
                    if num_of_clusters % len(self.categories) == 0:
                        return 0  # TODO:Add code for even cluster Number
                    else:  # sets the cluster as any other distribution
                        dataset["cluster"] = cluster
                        self.cfg[f"Node_{idx}"] = cluster
                else:
                    if num_of_clusters % len(self.categories) == 0:  # sets the cluster as any other distribution
                        dataset["cluster"] = cluster
                        self.cfg[f"Node_{idx}"] = cluster
                    else:
                        if idx < 20:  # TODO Hardcoded for 30 nodes and 3 clusters!!
                            cluster = idx % len(self.categories)
                        else:
                            cluster = len(self.categories)

            dataset["cluster"] = cluster
            self.cfg["clusters_for_nodes"][idx] = cluster


        info_datasets = {}
        for idx, dataset in enumerate(self.datasets): #creates metadata for dataset for sharing
            info_datasets[f"node{idx}"] = dataset.loc[dataset["train"] == "train"][self.categories].sum().to_dict()

        if meet_other_nodes:
            counter = {}
            for i in range(len(self.datasets)):
                counter[f"node{i}"] = []
            for i in range(len(self.datasets)):
                shared = random.sample(range(len(self.datasets)), datasets_seen)
                if i in shared:
                    shared.remove(i)
                counter[f"node{i}"] += shared #adds all shared nodes to node i
                for shared_node in shared:
                    counter[f"node{shared_node}"].append(i) #adds the sharing node to other nodes
            for key in counter.keys():
                counter[key] = list(set(counter[key]))
                random.shuffle(counter[key])

        for idx, dataset in enumerate(self.datasets):
            if meet_other_nodes:
                shared_nodes = counter[f"node{idx}"]
            else:
                cluster = dataset["cluster"].drop_duplicates()[0]
                shared_nodes = []
                for key, value in self.cfg["clusters_for_nodes"].items():
                    if value == cluster:
                        shared_nodes.append(key)
            info_dataset = copy.deepcopy(info_datasets[f"node{idx}"])
            if idx in shared_nodes:
                shared_nodes.remove(idx)
            for shared_node in shared_nodes:
                needed_category = min(info_dataset, key=lambda k: info_dataset[k]) #+calculates needed category
                shared_dataset = copy.deepcopy(self.datasets[shared_node])
                shared_dataset = shared_dataset.loc[shared_dataset["train"] == "train"]
                shared_dataset = shared_dataset.loc[shared_dataset["node"] == f"{shared_node}"] #kicks out shared data to reduce redundancy
                shared_dataset.sort_values(by=needed_category, inplace=True, ascending=False)
                shared_dataset = shared_dataset.iloc[0:int(len(shared_dataset) * shared_data)]
                self.datasets[idx] = pd.concat([self.datasets[idx],shared_dataset])
                for category in self.categories:
                    info_dataset[category] += shared_dataset[category].sum()

        for idx, dataset in enumerate(self.datasets):
            dataset["node"] = idx
            dataset_train = dataset.loc[dataset["train"] == "train"]
            dataset_val = dataset.loc[dataset["train"] == "val"]
            dataset_train["path_to_image"] = files.loc[files["filename"].isin(dataset_train["filename"])][0].values  # gets all images
            dataset_val["path_to_image"] = files.loc[files["filename"].isin(dataset_val["filename"])][0].values  # gets all images
            self.datasets[idx] = pd.concat([dataset_train, dataset_val])
            self.datasets[idx].apply(func=copy_file, axis=1)

    def copy_files_for_ensemble(self, num_of_clusters:int,path_to_dataset:str, datasharing:bool=False, shared:float=0,mix_cluster:bool=None, one_sided_dirichlet:bool=False) -> None:

        if datasharing:
            self.cfg["output_path"] = os.path.join(self.cfg["output_path"], "ensemble_datasharing", self.cfg["distribution"],f"shared{shared}")
            if os.path.isdir(self.cfg["output_path"]):
                raise Exception("Directory already exist use a different output_path for this distribution")
            self.cfg["shared"] = shared
            self.cfg["Datasharing"] = True
            if shared <= 0 or shared > 1:
                raise Exception("Shared amount too big or not set!")
        else:
            self.cfg["output_path"] = os.path.join(self.cfg["output_path"], self.cfg["distribution"])
            if os.path.isdir(os.path.join(self.cfg["output_path"], self.cfg["distribution"])):
                raise Exception("Directory already exist use a different output_path for this distribution")
            self.cfg["Datasharing"] = False
        self.cfg["num_of_clusters"] = num_of_clusters
        self.cfg["path_to_dataset"] = path_to_dataset


        files = pd.DataFrame(glob(os.path.join(path_to_dataset, "**", "*")))
        files["filename"] = files[0].apply(os.path.basename)
        files["filename"] = files["filename"].apply(lambda x: x[:-4])

        data_without_scenes = self.data.loc[~(self.data["scene"] == self.data["name"])]
        for idx, dataset in tqdm(enumerate(self.datasets), desc=f"Copying data for Nodes:"):
            #os.makedirs(os.path.join(self.cfg["output_path"], str(idx), "images", "train"))
            #os.makedirs(os.path.join(self.cfg["output_path"], str(idx), "images", "val"))
            #os.makedirs(os.path.join(self.cfg["output_path"], str(idx), "labels", "train"))
            #os.makedirs(os.path.join(self.cfg["output_path"], str(idx), "labels", "val"))
            dataset = data_without_scenes.loc[data_without_scenes["scene"].isin(dataset["scene"])]
            dataset["filename"] = dataset["name"].apply(os.path.basename)
            dataset["filename"] = dataset["filename"].apply(lambda x: x[:-4])
            dataset["output_path"] = self.cfg["output_path"]
            dataset["distribution"] = self.cfg["distribution"]
            dataset["node"] = str(idx)
            dataset_train = dataset.sample(frac=0.8, random_state=self.cfg["random_state"])
            dataset_val = dataset.drop(dataset_train.index)
            dataset_train["path_to_image"] = files.loc[files["filename"].isin(dataset_train["filename"])][0].values  #gets all images
            dataset_val["path_to_image"] = files.loc[files["filename"].isin(dataset_val["filename"])][0].values  #gets all images
            dataset_train["train"] = "train"
            dataset_val["train"] = "val"
            dataset = pd.concat([dataset_train, dataset_val])
            self.datasets[idx] = dataset
            #dataset.apply(func=copy_files, axis=1)

        self.shared = pd.DataFrame()
        for idx, dataset in enumerate(self.datasets):
            cluster = idx % num_of_clusters
            if self.cfg["distribution"] == "dirichlet":
                if mix_cluster == None:
                    raise Exception("Paramater mix_clusters is required for dirichlet distribution!")
                elif mix_cluster:
                    if num_of_clusters % len(self.categories) == 0:
                        return 0 #TODO:Add code for even cluster Number
                    else: #sets the cluster as any other distribution
                        dataset["cluster"] = cluster
                        self.cfg[f"Node_{idx}"] = cluster
                else:
                    if num_of_clusters % len(self.categories) == 0: #sets the cluster as any other distribution
                        dataset["cluster"] = cluster
                        self.cfg[f"Node_{idx}"] = cluster
                    else:
                        if idx < 20: #TODO Hardcoded for 30 nodes and 3 clusters!!
                            cluster = idx % len(self.categories)
                        else:
                            cluster = len(self.categories)

            dataset["cluster"] = cluster
            self.cfg["clusters_for_nodes"] = {}
            self.cfg["clusters_for_nodes"][idx] = cluster





            shared_dataset_train = dataset.loc[dataset["train"] == "train"]
            shared_dataset_train = shared_dataset_train.drop("path_to_image", axis=1) #deletes wrong path
            shared_dataset_train = shared_dataset_train.sample(frac=(shared), random_state=self.cfg["random_state"])
            shared_dataset_val = dataset.loc[dataset["train"] == "val"]
            shared_dataset_val = shared_dataset_val.drop("path_to_image", axis=1)
            shared_dataset_val = shared_dataset_val.sample(frac=1, random_state=self.cfg["random_state"])
            shared_dataset_train["path_to_image"] = files.loc[files["filename"].isin(shared_dataset_train["filename"])][0].values
            shared_dataset_val["path_to_image"] = files.loc[files["filename"].isin(shared_dataset_val["filename"])][0].values
            shared_dataset_train["node"] = f"shared{cluster}"
            shared_dataset_val["node"] = f"shared{cluster}"
            self.shared = pd.concat([self.shared, shared_dataset_train, shared_dataset_val])
            self.datasets[idx] = dataset
        #for cluster in range(num_of_clusters):
            #os.makedirs(os.path.join(self.cfg["output_path"], f"shared{cluster}", "images", "train"))
            #os.makedirs(os.path.join(self.cfg["output_path"], f"shared{cluster}", "images", "val"))
            #os.makedirs(os.path.join(self.cfg["output_path"], f"shared{cluster}", "labels", "train"))
            #os.makedirs(os.path.join(self.cfg["output_path"], f"shared{cluster}", "labels", "val"))

        #self.shared.apply(func=copy_files, axis=1)

    def copy_files_federated(self, path_to_dataset:str, datasharing:bool=False, shared:float=0,one_sided_dirichlet:bool=False) -> None:
        self.cfg["one_sided"] = one_sided_dirichlet
        if datasharing:
            self.cfg["output_path"] = os.path.join(self.cfg["output_path"], "datasharing",self.cfg["distribution"], f"shared{shared}")
            if os.path.isdir(self.cfg["output_path"]):
                raise Exception("Directory already exist use a different output_path for this distribution")
            self.cfg["shared"] = shared
            self.cfg["Datasharing"] = True
            if shared <= 0 or shared > 1:
                raise Exception("Shared amount too big or not set!")
        else:
            if one_sided_dirichlet:
                self.cfg["output_path"] = os.path.join(self.cfg["output_path"],"oneclass_nodes" ,self.cfg["distribution"])
            else:
                self.cfg["output_path"] = os.path.join(self.cfg["output_path"], self.cfg["distribution"])
            if os.path.isdir(os.path.join(self.cfg["output_path"])):
                raise Exception("Directory already exist use a different output_path for this distribution")
            self.cfg["Datasharing"] = False


        self.cfg["path_to_dataset"] = path_to_dataset

        files = pd.DataFrame(glob(os.path.join(path_to_dataset, "**", "*")))
        files["filename"] = files[0].apply(os.path.basename)
        files["filename"] = files["filename"].apply(lambda x: x[:-4])


        data_without_scenes = self.data.loc[~(self.data["scene"] == self.data["name"])]
        for idx, dataset in tqdm(enumerate(self.datasets), desc=f"Copying data for Nodes:"):
            os.makedirs(os.path.join(self.cfg["output_path"], str(idx), "images", "train"))
            os.makedirs(os.path.join(self.cfg["output_path"], str(idx), "images", "val"))
            os.makedirs(os.path.join(self.cfg["output_path"], str(idx), "labels", "train"))
            os.makedirs(os.path.join(self.cfg["output_path"], str(idx), "labels", "val"))
            category = dataset["class"].drop_duplicates()
            if (len(category) > 1) & (self.cfg["distribution"] == "dirichlet"):
                raise Exception("Something is wrong with the dirichlet distribution")
            dataset = data_without_scenes.loc[data_without_scenes["scene"].isin(dataset["scene"])]
            if one_sided_dirichlet:
                category = self.categories.index(category[0])
                dataset["class"] = category
            dataset["filename"] = dataset["name"].apply(os.path.basename)
            dataset["filename"] = dataset["filename"].apply(lambda x: x[:-4])
            dataset["output_path"] = self.cfg["output_path"]
            dataset["distribution"] = self.cfg["distribution"]
            dataset["node"] = str(idx)
            dataset_train = dataset.sample(frac=0.8, random_state=self.cfg["random_state"])
            dataset_val = dataset.drop(dataset_train.index)
            dataset_train["path_to_image"] = files.loc[files["filename"].isin(dataset_train["filename"])][0].values
            dataset_val["path_to_image"] = files.loc[files["filename"].isin(dataset_val["filename"])][0].values
            dataset_train["train"] = "train"
            dataset_val["train"] = "val"
            dataset = pd.concat([dataset_train, dataset_val])
            self.datasets[idx] = dataset
            if one_sided_dirichlet:
                dataset.apply(func=copy_files, axis=1)
                #dataset.apply(func=copy_files_one_sided, axis=1)
            else:
                continue
                dataset.apply(func=copy_files, axis=1)
        if datasharing:
            self.shared = pd.DataFrame()
            for dataset in self.datasets:
                shared_dataset_train = dataset.loc[dataset["train"] == "train"]
                shared_dataset_train = shared_dataset_train.drop("path_to_image", axis=1)
                shared_dataset_train = shared_dataset_train.sample(frac=shared, random_state=self.cfg["random_state"])
                shared_dataset_val = dataset.loc[dataset["train"] == "val"]
                shared_dataset_val = shared_dataset_val.drop("path_to_image", axis=1)
                shared_dataset_val = shared_dataset_val.sample(frac=1, random_state=self.cfg["random_state"])
                shared_dataset_train["path_to_image"] = files.loc[files["filename"].isin(shared_dataset_train["filename"])][0].values
                shared_dataset_val["path_to_image"] = files.loc[files["filename"].isin(shared_dataset_val["filename"])][0].values
                self.shared = pd.concat([self.shared,shared_dataset_train, shared_dataset_val])
            os.makedirs(os.path.join(self.cfg["output_path"], "shared", "images", "train"))
            os.makedirs(os.path.join(self.cfg["output_path"], "shared", "images", "val"))
            os.makedirs(os.path.join(self.cfg["output_path"], "shared", "labels", "train"))
            os.makedirs(os.path.join(self.cfg["output_path"], "shared", "labels", "val"))
            self.shared["node"] = "shared"
            self.shared.apply(func=copy_files, axis=1)

    def copy_files_scenes(self, path_to_dataset: str, output_path: str,datasharing: bool = False) -> None:
        weather_cond = list(data.datasets.keys())
        if datasharing:
            return 0
            #if os.path.isdir(os.path.join(output_path, "datasharing", self.cfg["distribution"])):
            #    raise Exception("Directory already exist use a different output_path for this distribution")
        else:
            for weather in weather_cond:
                if os.path.isdir(os.path.join(output_path, self.cfg["distribution"], weather)):
                    raise Exception("Directory already exist use a different output_path for this distribution")

        self.cfg["output_path"] = output_path
        self.cfg["path_to_dataset"] = path_to_dataset
        files = pd.DataFrame(glob(os.path.join(path_to_dataset, "**", "*")))
        files["filename"] = files[0].apply(os.path.basename)
        files["filename"] = files["filename"].apply(lambda x: x[:-4])


        data_without_scenes = self.data.loc[~(self.data["scene"] == self.data["name"])]
        for weather in weather_cond:
            for idx, dataset_weather in tqdm(enumerate(self.datasets[weather]), desc=f"Copying data for Nodes:"):
                os.makedirs(os.path.join(self.cfg["output_path"], self.cfg["distribution"],weather, str(idx), "images", "train"))
                os.makedirs(os.path.join(self.cfg["output_path"], self.cfg["distribution"],weather, str(idx), "images", "val"))
                os.makedirs(os.path.join(self.cfg["output_path"], self.cfg["distribution"],weather, str(idx), "labels", "train"))
                os.makedirs(os.path.join(self.cfg["output_path"], self.cfg["distribution"],weather, str(idx), "labels", "val"))
                dataset = data_without_scenes.loc[data_without_scenes["scene"].isin(dataset_weather["scene"])]
                dataset["filename"] = dataset["name"].apply(os.path.basename)
                dataset["filename"] = dataset["filename"].apply(lambda x: x[:-4])
                dataset["output_path"] = self.cfg["output_path"]
                dataset["distribution"] = os.path.join(self.cfg["distribution"], weather)
                dataset["node"] = str(idx)
                dataset_train = dataset.sample(frac=0.80, random_state=self.cfg["random_state"])
                dataset_val = dataset.drop(dataset_train.index)
                dataset_train["path_to_image"] = files.loc[files["filename"].isin(dataset_train["filename"])][0].values  # gets all images
                dataset_val["path_to_image"] = files.loc[files["filename"].isin(dataset_val["filename"])][0].values  # gets all images
                dataset_train["train"] = "train"
                dataset_val["train"] = "val"
                dataset = pd.concat([dataset_train, dataset_val])
                self.datasets[weather][idx] = dataset
                dataset.apply(func=copy_file, axis=1)

    def copy_files_smaller(self,path_to_dataset:str, datasharing:bool=False,
                           shared:float=0,num_of_clusters:int =3,mix_classes:bool=False,
                           copy_files:bool=False, hetergenous:bool=False,
                           label_aware_datasharing:bool=False):

        self.cfg["one_sided"] = False  #change yaml creation to get one sided so this is not required TODO

        if hetergenous:
            self.cfg["distribution"] = f"{self.cfg['distribution']}_hetero"
        if label_aware_datasharing:
            self.cfg["distribution"] = f"{self.cfg['distribution']}_smartshared"
            self.cfg["output_path"] = os.path.join(self.cfg["output_path"], "small", "datasharing",
                                                   self.cfg["distribution"], f"shared{shared}")
            if os.path.isdir(self.cfg["output_path"]):
                raise Exception("Directory already exist use a different output_path for this distribution")
            self.cfg["shared"] = shared
            self.cfg["Datasharing"] = True
            if shared <= 0 or shared > 1:
                raise Exception("Shared amount too big or not set!")
        elif datasharing:
            self.cfg["output_path"] = os.path.join(self.cfg["output_path"],"small", "datasharing",self.cfg["distribution"], f"shared{shared}")
            if os.path.isdir(self.cfg["output_path"]):
                raise Exception("Directory already exist use a different output_path for this distribution")
            self.cfg["shared"] = shared
            self.cfg["Datasharing"] = True
            if shared <= 0 or shared > 1:
                raise Exception("Shared amount too big or not set!")
        else:
            self.cfg["output_path"] = os.path.join(self.cfg["output_path"],"small", self.cfg["distribution"])
            if os.path.isdir(os.path.join(self.cfg["output_path"])):
                raise Exception("Directory already exist use a different output_path for this distribution")
            self.cfg["Datasharing"] = False


        self.cfg["path_to_dataset"] = path_to_dataset

        files = pd.DataFrame(glob(os.path.join(path_to_dataset, "**", "*")))
        files["filename"] = files[0].apply(os.path.basename)
        files["filename"] = files["filename"].apply(lambda x: x[:-4])

        data_without_scenes = self.data.loc[~(self.data["scene"] == self.data["name"])]

        for idx, dataset in tqdm(enumerate(self.datasets), desc=f"Copying data for Nodes:"):
            if copy_files:
                os.makedirs(os.path.join(self.cfg["output_path"], str(idx), "images", "train"))
                os.makedirs(os.path.join(self.cfg["output_path"], str(idx), "images", "val"))
                os.makedirs(os.path.join(self.cfg["output_path"], str(idx), "labels", "train"))
                os.makedirs(os.path.join(self.cfg["output_path"], str(idx), "labels", "val"))
            if self.cfg["distribution"] == "dirichlet":
                category = dataset["class"].drop_duplicates()[0]
            dataset = copy.deepcopy(data_without_scenes.loc[data_without_scenes["scene"].isin(dataset["scene"])])
            dataset["filename"] = dataset["name"].apply(os.path.basename)
            dataset["filename"] = dataset["filename"].apply(lambda x: x[:-4])
            dataset["output_path"] = self.cfg["output_path"]
            dataset["distribution"] = self.cfg["distribution"]
            dataset["node"] = str(idx)
            dataset["sum_classes"] = dataset[self.categories].apply(func=sum, axis=1)
            if hetergenous:
                dataset = dataset[dataset[self.categories].apply(lambda x: (x > 0).any(), axis=1)] #take all labelled files
                if "ascending" not in locals():
                    ascending = True
                if "dirichlet" in self.cfg["distribution"]:
                    if "ascending_diri" not in locals():
                        ascending_diri = [True, True]
                if "dirichlet" in self.cfg["distribution"]:
                    ascender = (idx % len(self.categories))
                    dataset.sort_values(by=self.categories[ascender], ascending=ascending_diri[ascender], inplace=True)
                    dataset = dataset.iloc[:int(len(dataset) * 0.4)]
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
            dataset_train = dataset.sample(frac=0.8, random_state=self.cfg["random_state"])
            dataset_val = dataset.drop(dataset_train.index)
            dataset_train["path_to_image"] = files.loc[files["filename"].isin(dataset_train["filename"])][0].values  #gets all images
            dataset_val["path_to_image"] = files.loc[files["filename"].isin(dataset_val["filename"])][0].values  #gets all images
            dataset_train["train"] = "train"
            dataset_val["train"] = "val"
            dataset = pd.concat([dataset_train, dataset_val])
            self.datasets[idx] = dataset
            if copy_files and not label_aware_datasharing:
                dataset.apply(func=copy_file, axis=1)

        if label_aware_datasharing:
            self.cfg["clusters_for_nodes"] = {}
            for idx, dataset in enumerate(self.datasets):
                if idx < 12:
                    cluster = idx % len(data.categories)
                else:
                    cluster = len(data.categories)
                dataset["cluster"] = cluster
                self.cfg["clusters_for_nodes"][idx] = cluster


            info_datasets = {}
            for idx, dataset in enumerate(self.datasets):  # creates metadata for dataset for sharing
                info_datasets[f"node{idx}"] = dataset.loc[dataset["train"] == "train"][self.categories].sum().to_dict()


            for idx, dataset in enumerate(self.datasets):
                cluster = dataset["cluster"].drop_duplicates()[0]
                shared_nodes = []
                for key, value in self.cfg["clusters_for_nodes"].items():
                    if value == cluster:
                        shared_nodes.append(key)
                info_dataset = copy.deepcopy(info_datasets[f"node{idx}"])
                if idx in shared_nodes:
                    shared_nodes.remove(idx)
                for shared_node in shared_nodes:
                    needed_category = min(info_dataset, key=lambda k: info_dataset[k])  #calculates needed category
                    shared_dataset = copy.deepcopy(self.datasets[shared_node])
                    shared_dataset = shared_dataset.loc[shared_dataset["train"] == "train"]
                    shared_dataset = shared_dataset.loc[shared_dataset["node"] == f"{shared_node}"]  # kicks out shared data to reduce redundancy
                    shared_dataset.sort_values(by=needed_category, inplace=True, ascending=False)
                    shared_dataset = shared_dataset.iloc[0:int(len(shared_dataset) * shared)]
                    self.datasets[idx] = pd.concat([self.datasets[idx], shared_dataset])
                    for category in self.categories:
                        info_dataset[category] += shared_dataset[category].sum()

            for idx, dataset in enumerate(self.datasets):
                dataset["node"] = f"{idx}"
                dataset_train = dataset.loc[dataset["train"] == "train"]
                dataset_val = dataset.loc[dataset["train"] == "val"]
                dataset_train["path_to_image"] = files.loc[files["filename"].isin(dataset_train["filename"])][0].values  # gets all images
                dataset_val["path_to_image"] = files.loc[files["filename"].isin(dataset_val["filename"])][0].values  # gets all images
                self.datasets[idx] = pd.concat([dataset_train, dataset_val])
                if copy_files:
                    self.datasets[idx].apply(func=copy_file, axis=1)


        elif datasharing and mix_classes:
            if mix_classes and self.cfg["distribution"] == "dirichlet":
                return 0

        elif datasharing:
            self.shared = pd.DataFrame()
            self.cfg["Datasets"] = {}
            for idx, dataset in enumerate(self.datasets):
                if idx < 12:
                    cluster = idx % len(self.categories)
                else:
                    cluster = len(self.categories)
                shared_dataset_train = dataset.loc[dataset["train"] == "train"]
                shared_dataset_train = shared_dataset_train.drop("path_to_image", axis=1)
                shared_dataset_train = shared_dataset_train.sample(frac=shared, random_state=self.cfg["random_state"])
                shared_dataset_train["path_to_image"] = files.loc[files["filename"].isin(shared_dataset_train["filename"])][0].values
                shared_dataset_train["node"] = f"shared{cluster}"
                self.shared = pd.concat([self.shared, shared_dataset_train])
            for idx, dataset in enumerate(self.datasets):
                if idx < 12: #HARD CODED TODO:find uniform solution?
                    cluster = idx % len(self.categories)
                else:
                    cluster = len(self.categories)
                self.datasets[idx] = pd.concat([self.datasets[idx],copy.deepcopy(self.shared.loc[self.shared["node"] == f"shared{cluster}"])])
                self.datasets[idx]["cluster"] = cluster
            if copy_files:
                for i in range(num_of_clusters):
                    os.makedirs(os.path.join(self.cfg["output_path"], f"shared{i}", "images", "train"))
                    #os.makedirs(os.path.join(self.cfg["output_path"], f"shared{i}", "images", "val")) not required
                    os.makedirs(os.path.join(self.cfg["output_path"], f"shared{i}", "labels", "train"))
                    #os.makedirs(os.path.join(self.cfg["output_path"], f"shared{i}", "labels", "val")) not required
            if copy_files:
                self.shared.apply(func=copy_file, axis=1)



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


    def copy_files_centrelazied(self, path_to_dataset: str, output_path, create_val:bool = True) ->None:
        output_path = os.path.join(output_path)
        if os.path.isdir(output_path):
            shutil.rmtree(os.path.join(output_path))
        os.mkdir(output_path)
        os.mkdir(os.path.join(output_path,"images"))
        os.mkdir(os.path.join(output_path,"images", "train"))
        os.mkdir(os.path.join(output_path,"labels"))
        os.mkdir(os.path.join(output_path,"labels", "train"))
        if create_val:
            os.mkdir(os.path.join(output_path, "images", "val"))
            os.mkdir(os.path.join(output_path, "labels", "val"))

        train_dataset = pd.DataFrame()
        val_dataset = pd.DataFrame()
        scenes = self.data["scene"].drop_duplicates()
        all_data = copy.deepcopy(self.data)
        all_data = all_data.loc[~(all_data["scene"] == all_data["name"])]
        for scene in tqdm(scenes, desc="Create Train and Validation Set"):
            train_dataset_scene = all_data.loc[all_data["scene"] == scene].sample(frac=0.8, random_state=self.cfg["random_state"])
            val_dataset_scene = all_data.loc[all_data["scene"] == scene].drop(train_dataset_scene.index)
            train_dataset = pd.concat([train_dataset, train_dataset_scene])
            val_dataset = pd.concat([val_dataset, val_dataset_scene])


        train_dataset["filename"] = train_dataset["name"].apply(os.path.basename)
        val_dataset["filename"] = val_dataset["name"].apply(os.path.basename)
        train_dataset["filename"] = train_dataset["filename"].apply(lambda x: x[:-4])
        val_dataset["filename"] = val_dataset["filename"].apply(lambda x: x[:-4])
        files = pd.DataFrame(glob(os.path.join(path_to_dataset,"**","*")))
        files[0] = files[0].apply(os.path.join)
        files["filename"] = files[0].apply(os.path.basename)
        files["filename"] = files["filename"].apply(lambda x: x[:-4])

        files_copied = files.loc[files["filename"].isin(train_dataset["filename"])]
        files_copied.apply(func=copy_images_train, axis=1)
        files_copied = files.loc[files["filename"].isin(val_dataset["filename"])]
        files_copied.apply(func=copy_images_val, axis=1)
        train_dataset.apply(func=copy_txt_train, axis=1)
        val_dataset.apply(func=copy_txt_val, axis=1)





    def create_yaml(self, path_to_yaml:str):
        self.cfg["yaml_files"] = os.path.join(path_to_yaml, self.cfg["distribution"])
        if self.cfg["Datasharing"]:
            self.cfg["yaml_files"] = os.path.join(path_to_yaml, "datasharing", self.cfg["distribution"], f"shared{self.cfg['shared']}")
        if os.path.isdir(self.cfg["yaml_files"]):
            raise Exception("path to yaml already exists check if the Yaml are needed first")
        os.makedirs(self.cfg["yaml_files"])
        for idx, dataset in enumerate(self.datasets):
            with open(os.path.join(self.cfg["yaml_files"], f"{idx}.yaml"), "w") as f:
                f.write(f"path: {self.cfg['output_path']}\n")  # writes the
                f.write(f"train: {idx}/images/train\n")
                f.write(f"val: {idx}/images/val\n")
                f.write("names:\n")
                f.write("  0: human.pedestrian\n")
                f.write("  1: vehicle.car\n")

        with open(os.path.join(self.cfg["yaml_files"], f"all.yaml"), "w") as f: # writes the all.yaml file
            f.write(f"path: {self.cfg['output_path']}\n")
            f.write(f"train: [0/images/train")
            for idx in range(1,self.cfg['num_of_datasets']):
                f.write(f"{idx}/images/train,")
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


    def create_yaml_for_datasharing(self, path_to_yaml:str):
        self.cfg["yaml_files"] = os.path.join(path_to_yaml, self.cfg["distribution"])
        if self.cfg["Datasharing"]:
            self.cfg["yaml_files"] = os.path.join(path_to_yaml, "datasharing", self.cfg["distribution"], f"shared{self.cfg['shared']}")
        if os.path.isdir(self.cfg["yaml_files"]):
            raise Exception("path to yaml already exists check if the Yaml are needed first")
        os.makedirs(self.cfg["yaml_files"])
        for idx, dataset in enumerate(self.datasets):
            cluster = dataset["cluster"][0]
            with open(os.path.join(self.cfg["yaml_files"], f"{idx}.yaml"), "w") as f:
                f.write(f"path: {self.cfg['output_path']}\n")  # writes the
                f.write(f"train: [{idx}/images/train, shared{cluster}/images/train]\n")
                f.write(f"val: {idx}/images/val\n")
                f.write("names:\n")
                f.write("  0: human.pedestrian\n")
                f.write("  1: vehicle.car\n")
        with open(os.path.join(self.cfg["output_path"], "config.json"), "w") as outfile:
            json.dump(self.cfg, outfile, indent=4)



def copy_files_one_sided(row):
    output_path = os.path.join(row["output_path"], str(row["node"]))
    shutil.copy2(row["path_to_image"], os.path.join(output_path, "images", row["train"]))
    name = f"{row['filename']}.txt"
    lines = []
    with open(row["name"], "r") as orig:
        for line in orig.readlines():
            if int(line[0]) == int(row["class"]):
                lines.append(line)
    with open(os.path.join(output_path, "labels", row["train"],name), "w") as out:
        out.writelines(lines)





def copy_file(row):
    output_path = os.path.join(row["output_path"], str(row["node"]))
    shutil.copy2(row["name"], os.path.join(output_path, "labels", row["train"]))
    shutil.copy2(row["path_to_image"], os.path.join(output_path, "images", row["train"]))

def copy_images_train(row):
    path = Path().resolve()
    value1 = row[0]
    value2 = row["filename"]
    path = os.path.join("/work/scratch/td38heni/Centralized","images","train")


    shutil.copy2(value1, path)

def copy_images_val(row):
    path = Path().resolve()
    value1 = row[0]
    value2 = row["filename"]
    path = os.path.join("/work/scratch/td38heni/Centralized","images","val")

    # Perform your operations on the values
    shutil.copy2(value1, path)   # Replace with your desired operations

def copy_txt_train(row):
    path = Path().resolve()
    value1 = row["name"]
    value2 = row["filename"]
    path = os.path.join("/work/scratch/td38heni/Centralized", "labels", "train")

    # Perform your operations on the values
    shutil.copy2(value1, path)   # Replace with your desired operations

def copy_txt_val(row):
    path = Path().resolve()
    value1 = row["name"]
    value2 = row["filename"]
    path = os.path.join("/work/scratch/td38heni/Centralized","labels","val")

    # Perform your operations on the values
    shutil.copy2(value1, path)  # Replace with your desired operations


