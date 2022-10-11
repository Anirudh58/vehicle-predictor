import os

import matplotlib.pyplot as plt
import numpy as np

import torch


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
