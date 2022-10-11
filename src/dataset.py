import os

import numpy as np
import pandas as pd

import torch
import torchvision

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class VehiclePredictorDataset(Dataset):
    """
    Custom dataset for VMMRdb
    """

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.directories = os.listdir(root_dir)

        # image paths
        self.images = []

        # utility maps
        self.class_to_idx = {}
        self.idx_to_class = {}
        self.make_to_idx = {}
        self.idx_to_make = {}
        self.model_to_idx = {}
        self.idx_to_model = {}
        self.year_to_idx = {}
        self.idx_to_year = {}

        # for stats
        self.make_counts = {}
        self.make_model_counts = {}
        self.make_model_year_counts = {}
        self.year_counts = {}

        for i, directory in enumerate(self.directories):

            # the entire class name
            self.class_to_idx[directory] = i
            self.idx_to_class[i] = directory

            # the make name
            make = directory.split('_')[0]
            if make not in self.make_to_idx:
                self.make_to_idx[make] = len(self.make_to_idx)
                self.idx_to_make[len(self.idx_to_make)] = make

            # the model name
            model = "".join(directory.split('_')[1:-1])
            if model not in self.model_to_idx:
                self.model_to_idx[model] = len(self.model_to_idx)
                self.idx_to_model[len(self.idx_to_model)] = model

            # the year
            year = directory.split('_')[-1]
            if year not in self.year_to_idx:
                self.year_to_idx[year] = len(self.year_to_idx)
                self.idx_to_year[len(self.idx_to_year)] = year

            # iterate through all the images in the directory
            for image in os.listdir(os.path.join(root_dir, directory)):
                self.images.append(os.path.join(root_dir, directory, image))

                if make not in self.make_counts:
                    self.make_counts[make] = 1
                else:
                    self.make_counts[make] += 1

                if make + '_' + model not in self.make_model_counts:
                    self.make_model_counts[make + '_' + model] = 1
                else:
                    self.make_model_counts[f"{make}_{model}"] += 1

                if make + '_' + model + '_' + year not in self.make_model_year_counts:
                    self.make_model_year_counts[make + '_' + model + '_' + year] = 1
                else:
                    self.make_model_year_counts[f"{make}_{model}_{year}"] += 1

                if year not in self.year_counts:
                    self.year_counts[year] = 1
                else:
                    self.year_counts[year] += 1


    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        target = {}

        # get the class name
        class_name = image.split('/')[-2]
        target['class'] = self.class_to_idx[class_name]

        # get the make name
        make = class_name.split('_')[0]
        target['make'] = self.make_to_idx[make]

        # get the model name
        model = "".join(class_name.split('_')[1:-1])
        target['model'] = self.model_to_idx[model]

        # get the year
        year = class_name.split('_')[-1]
        target['year'] = self.year_to_idx[year]

        # load the image
        image = torchvision.io.read_image(image)

        if self.transform:
            image = self.transform(image)

        return image, target
                