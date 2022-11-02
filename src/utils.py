import copy
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import sklearn
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

import torch
import torchvision
from torchvision import transforms
import torch.nn as nn

sys.path.append('./../')
from src.dataset import VehiclePredictorDataset


# Global constants
root_path = './../'
data_path = os.path.join(root_path, 'data')
dataset_path = os.path.join(data_path, 'VMMRdb')
with open(os.path.join(data_path, 'make_model_most_common_100.pkl'), 'rb') as f:
    target_make_model_labels = pickle.load(f)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
vp_dataset = VehiclePredictorDataset(root_dir=dataset_path, target_make_model_labels=target_make_model_labels)
num_images = len(vp_dataset)
num_labels = len(vp_dataset.make_model_counts)
class_distribution = vp_dataset.make_model_counts


def plot_images(images, titles=None):
    """
    plot a list of images given the image tensors
    """

    # convert to numpy
    images = [image.numpy() for image in images]

    cols = 5
    rows = len(images) // cols + 1

    fig = plt.figure(figsize=(25, 25))

    for i, image in enumerate(images):
        fig.add_subplot(rows, cols, i + 1)
        plt.imshow(np.transpose(image, (1, 2, 0)))
        plt.axis('off')
        if titles is not None:
            plt.title(titles[i])

def get_model(num_classes, backbone_model):
    
    if backbone_model == 'resnet18':
        model = torchvision.models.resnet18(weights='ResNet18_Weights.DEFAULT')
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    if backbone_model == 'mobilenet_v2':
        model = torchvision.models.mobilenet_v2(weights='MobileNetV2_Weights.DEFAULT')
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)

    if backbone_model == 'resnet50':
        model = torchvision.models.resnet50(weights='ResNet50_Weights.DEFAULT')
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    
    return model

def train_model(model, model_info, dataset_sizes, dataloader_train, dataloader_val, criterion, optimizer, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs-1}")
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:

            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in tqdm(dataloader_train if phase == 'train' else dataloader_val, position=0, leave=True):

                inputs = inputs.to(device)
                labels = labels['make_model'].to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            time_elapsed = time.time() - since
            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} Time: {time_elapsed:.0f}m {time_elapsed:.0f}s")

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best val Acc: {best_acc:4f}")

    # save best model weights
    model_name = "_".join([model_info[key] for key in model_info])
    torch.save(best_model_wts, os.path.join(root_path, 'models', f"{model_name}.pth"))


def evaluate_model(model, dataset_sizes, dataloader_test):
    model.eval()
    running_corrects = 0

    # confusion matrix
    y_true = []
    y_pred = []
    for inputs, labels in tqdm(dataloader_test, position=0, leave=True):
        inputs = inputs.to(device)
        labels = labels['make_model'].to(device)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        running_corrects += torch.sum(preds == labels.data)

        y_true.extend(labels.tolist())
        y_pred.extend(preds.tolist())

    test_acc = running_corrects.double() / dataset_sizes['test']
    print(f"Test Acc: {test_acc:.4f}")

    # confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    return cm