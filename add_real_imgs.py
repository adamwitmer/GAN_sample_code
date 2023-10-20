from __future__ import division
from setproctitle import setproctitle
import csv
import argparse
import os
import torch
import shutil
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
import torch.optim
import torch.utils.data
import torchvision.utils as utils
import torch.utils.data.sampler as sampler
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.nn import DataParallel
# from torchsample.callbacks import EarlyStopping
# import torchsample.transforms as tstransforms

from HESCnet import HESCnet


import time
import random
import sys
import numpy as np
import pdb
from PIL import Image
from collections import defaultdict
import pdb
import csv
import cv2
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, draw, show, clf
import pdb
import progressbar
try:
    import accimage
except ImportError:
    accimage = None

# load training data based on random seeds
# save images into respective folders with synthetic images
class Sampler(object):
    """Base class for all Samplers.
    Every Sampler subclass has to provide an __iter__ method, providing a way
    to iterate over indices of dataset elements, and a __len__ method that
    returns the length of the returned iterators.
    """

    def __init__(self, data_source):
        pass

    def __iter__(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class SubsetSampler(Sampler):
    """Samples elements randomly from a given list of indices, without replacement.
    Arguments:
        indices (list): a list of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)


randseeds = []

save_folder = ''

classes = ['Debris', 'Dense', 'Diff', 'Spread']

folders = [''] 

base_data_folder = ''

for folder in folders:

    for i, seed in enumerate(randseeds):  # 5 random seeds/folders/fold validation

        new_save_folder = os.path.join(save_folder, folder, 'fold_{}'.format(i))

        for clss in classes:
        
            train_save_folder = '{}/train/{}'.format(new_save_folder, clss)
            os.makedirs(train_save_folder, exist_ok=True)
            test_save_folder = '{}/test/{}'.format(new_save_folder, clss)
            os.makedirs(test_save_folder, exist_ok=True)
            # valid_save_folder = '{}/valid/{}'.format(new_save_folder, clss)
            # os.makedirs(valid_save_folder, exist_ok=True)

        train_data = datasets.ImageFolder(base_data_folder, transform=transforms.ToTensor())  #  os.path.join( , folder)

        # split testing set indices
        num_train = len(train_data)
        indices = list(range(num_train))
        split = int(np.floor(0.2 * num_train))  # 0.2 --> 80 % training; 0.8 --> 20% training data
        np.random.seed(seed)
        np.random.shuffle(indices)
        # train_idx, valid_idx, test_idx = indices[split:], indices[:int(split / 2)], indices[int(split / 2):split]
        train_idx, test_idx = indices[split:], indices[:split]

        img_list = train_data.imgs

        for k, idx in enumerate(train_idx):

            # output status
            sys.stdout.write('\r Training Folder: {}, Fold:{}, Moving images: {}/{}'
                             .format(folder, i+1, k+1, len(train_idx)))

            # gather image path and class
            img_path = img_list[idx][0]
            img_name = os.path.basename(img_path)
            img_class = classes[img_list[idx][1]]

            
            # move file from source to destination
            dest_file = os.path.join(new_save_folder, 'train', img_class, img_name)
            shutil.copy(img_path, dest_file)

        for k, idx in enumerate(test_idx):

            # output status
            sys.stdout.write('\rTesting Folder: {}, Fold:{}, Moving images: {}/{}'
                             .format(folder, i + 1, k + 1, len(test_idx)))

            # gather image path and class
            img_path = img_list[idx][0]
            img_name = os.path.basename(img_path)
            img_class = classes[img_list[idx][1]]

            dest_file = os.path.join(new_save_folder, 'test', img_class, img_name)
            shutil.copy(img_path, dest_file)


        # for k, idx in enumerate(valid_idx):
        #     # output status
        #     sys.stdout.write('\rTesting Folder: {}, Fold:{}, Moving images: {}/{}'
        #                      .format(folder, i + 1, k + 1, len(test_idx)))

        #     # gather image path and class
        #     img_path = img_list[idx][0]
        #     img_name = os.path.basename(img_path)
        #     img_class = classes[img_list[idx][1]]

        #     dest_file = os.path.join(new_save_folder, 'valid', img_class, img_name)
        #     shutil.copy(img_path, dest_file)












