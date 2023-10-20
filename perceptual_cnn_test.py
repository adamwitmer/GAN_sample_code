from __future__ import division
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.utils.data.sampler as sampler
import torch.nn.functional as F
from torch.nn import DataParallel
from torchvision import datasets, models
# from tensorboardX import SummaryWriter
from torch.autograd import Variable
# import skimage.feature as sk
# from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
import scipy.misc
from scipy import stats
import argparse
import pickle
import random
from CustomDataset import CustomDataset, CustomDataset224
from ImbalancedSampler import ImbalancedDatasetSampler
import time
from datetime import timedelta
import pdb
import sys
import csv
import os
import cv2
from PIL import Image, ImageFilter
import numpy as np
# from setproctitle import setproctitle
from tessa import dictionary, sfta, chog
import matplotlib
from HESCnet import HESCnet
import vgg_models
from ShallowNet import ShallowNet
matplotlib.use('Agg')
import matplotlib.pyplot as plt
try:
    import accimage
except ImportError:
    accimage = None

class SmallScale(object):
    """Resize the input PIL Image to the given size.
    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be scaled.
        Returns:
            PIL Image: Rescaled image.
        """
        height, width = img.size
        if height < 64 or width < 64:
            return resize(img, self.size, self.interpolation)
        else:
            return img


def resize(img, size, interpolation=Image.BILINEAR):
    """Resize the input PIL Image to the given size.
    Args:
        img (PIL Image): Image to be resized.
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), the output size will be matched to this. If size is an int,
            the smaller edge of the image will be matched to this number maintaing
            the aspect ratio. i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    Returns:
        PIL Image: Resized image.
    """
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))
    if not (isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)):
        raise TypeError('Got inappropriate size arg: {}'.format(size))

    if isinstance(size, int):
        w, h = img.size
        if (w <= h and w == size) or (h <= w and h == size):
            return img
        if w < h:
            ow = size
            oh = int(size * h / w)
            return img.resize((ow, oh), interpolation)
        else:
            oh = size
            ow = int(size * w / h)
            return img.resize((ow, oh), interpolation)
    else:
        return img.resize(size[::-1], interpolation)


def _is_pil_image(img):
    if accimage is not None:
        return isinstance(img, (Image.Image, accimage.Image))
    else:
        return isinstance(img, Image.Image)


def calculate_metrics(confmat, n_class):

    tpr_cm = np.zeros(n_class)
    tnr_cm = np.zeros(n_class)
    f1_cm = np.zeros(n_class)
    # determine accuracy metrics
    for y in range(n_class):
        # true positive rate
        pos = np.sum(confmat[:, y])  # number of instances in the positive class
        tp = confmat[y, y]  # correctly classified positive incidents
        fp = np.sum(confmat[y, :]) - tp  # incorrectly classified negative instances
        tpr = tp / pos  # true positive classification rate
        tpr_cm[y] = tpr
        # true negative rate
        tn = np.trace(confmat) - tp  # correctly classified negative instances
        tnr = tn / (tn + fp)  # true negative rate
        tnr_cm[y] = tnr
        # f1 score
        ppv = tp / (tp + fp)  # positve prediction value
        f1 = 2 * ((ppv * tpr) / (ppv + tpr))  # f1 score
        f1_cm[y] = f1
        # dice similarity coefficient (dsc)
        # dsc = 2 * tp / (2 * tp + fp + tn)
        # dsc_cm[fold, y] = dsc

    return tpr_cm, tnr_cm, f1_cm


def build_dataset(train_path, fld, size):

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # for testing use first seed, CHANGE FOR CROSS VALIDATION to randseeds[net]
    randseeds = [1046, 6401, 51, 200589, 50098, 24568, 249, 7899, 2, 89000]

    custom_data = datasets.ImageFolder(train_path, transforms.Compose([SmallScale(64),
                                                                       transforms.Grayscale(),
                                                                       transforms.CenterCrop(64),
                                                                       # transforms.RandomHorizontalFlip(),
                                                                       # transforms.RandomVerticalFlip(),
                                                                       transforms.ToTensor(),
                                                                       normalize]))

    weight_data = datasets.ImageFolder(train_path, transform=transforms.Compose([transforms.Resize((224, 224)),
                                                                                 transforms.Grayscale(),
                                                                                 transforms.ToTensor()]
                                                                                )
                                       )

    num_train = len(custom_data)
    indices = list(range(num_train))
    split = int(np.floor(0.2 * num_train))

    np.random.seed(randseeds[fld])  # --> randseeds[net]
    np.random.shuffle(indices)

    train_idx, valid_idx, test_idx = indices[split:], indices[split//2:split], indices[:split//2]

    train_sampler = sampler.SubsetRandomSampler(train_idx)
    valid_sampler = sampler.SubsetRandomSampler(valid_idx)
    test_sampler = sampler.SubsetRandomSampler(test_idx)

    train_loader = torch.utils.data.DataLoader(custom_data,
                                               batch_size=32,
                                               sampler=train_sampler,
                                               shuffle=False,
                                               drop_last=True)

    valid_loader = torch.utils.data.DataLoader(custom_data,
                                               batch_size=32,
                                               sampler=valid_sampler,
                                               shuffle=False,
                                               drop_last=True)

    test_loader = torch.utils.data.DataLoader(custom_data,
                                              batch_size=32,
                                              sampler=test_sampler,
                                              shuffle=False,
                                              drop_last=False)

    weight_loader = torch.utils.data.DataLoader(weight_data,
                                                batch_size=128,
                                                sampler=train_sampler,
                                                shuffle=False)

    return train_loader, valid_loader, weight_loader, test_loader, test_idx


def take_random_patches(img, crop_size):

    # check image dimensions and resize
    # normalize loaded image
    img = (img - img.min()) / (img.max() - img.min())
    img = (img - 0.485) / 0.229
    if isinstance(crop_size, int):
        w, h = img.shape
        if w < h:
            ow = crop_size
            oh = int(crop_size * h / w)
            img.resize((ow, oh))
        else:
            oh = crop_size
            ow = int(crop_size * w / h)
            img.resize((ow, oh))
    else:
        img.resize(crop_size[::-1])

    # take 5 random crops from image and return as tensor
    crop_tensor = np.zeros([5, 1, 64, 64])
    w, h = img.shape
    th = tw = crop_size
    for crop in range(5):
        # take random crops and add to
        if w == tw and h == th:
            crop_tensor[crop, 0, :, :] = img[0:w + crop_size, 0:h + crop_size]
        else:
            x = random.randint(0, h - th)
            y = random.randint(0, w - tw)
            crop_tensor[crop, 0, :, :] = img[y:y+crop_size, x:x+crop_size]
    return torch.Tensor(crop_tensor)

# set parameters
fold = 0
img_size = 64

configs = ['']

# load network and dataset
seeds = [1046, 6401, 51, 200589, 50098, 24568, 249, 7899, 2, 89000]

data_path = ''
n_crossfolds = 1
n_classes = 4

for config in configs:

    tpr_cm = np.zeros([n_crossfolds, n_classes])  # True positive rate --> sensitivity
    tnr_cm = np.zeros([n_crossfolds, n_classes])  # true negative rate --> specificity
    f1_cm = np.zeros([n_crossfolds, n_classes])  # F1 score

    for fold in range(n_crossfolds):

        test_data_path = os.path.join(data_path, config, 'test')

        test_data = datasets.ImageFolder(test_data_path, transforms.Compose([SmallScale(64),
                                                                             transforms.Grayscale(),
                                                                             transforms.CenterCrop(64),
                                                                             # transforms.Resize(224),
                                                                             # transforms.RandomHorizontalFlip(),
                                                                             # transforms.RandomVerticalFlip(),
                                                                             transforms.ToTensor(),
                                                                             transforms.Normalize(
                                                                                 mean=[0.485, 0.456, 0.406],
                                                                                 std=[0.229, 0.224, 0.225])]))

        
        test_loader = torch.utils.data.DataLoader(test_data,
                                                   # sampler=test_sampler,
                                                   batch_size=32,
                                                   shuffle=False)


        predictions = []
        targets = []

        
        base_cnn_path = data_path 
        cnn_path = '{}/save/vgg19_{}_{}'.format(base_cnn_path, config, fold)
        save_path = os.path.join(cnn_path, 'data')  
        os.makedirs(save_path, exist_ok=True)
        model_path = os.path.join(cnn_path, 'trained_nn.pth')
        model = torch.load(model_path)
        model = model.cuda()
        model.eval()

        # get image predictions for entire test set
        for i, (input, target) in enumerate(test_loader):

            sys.stdout.write('\rConfiguration: {}, Fold: {}, Testing Batch {}/{}'
                             .format(config, fold+1, i, len(test_loader)))
            targets.append(target)
            target = target.cuda(async=True)
            input = input.cuda(async=True)
            input_var = Variable(input, volatile=True)
            target_var = Variable(target, volatile=True)
            output = model.forward(input_var)  # two column vector size of minibatch
            output_val = nn.functional.softmax(output).data.cpu().numpy()  # softmax output values
            output_pred = np.argmax(output_val, axis=1)

            predictions.append(output_pred)

        predictions = np.concatenate(predictions, axis=0)
        targets = np.concatenate(targets, axis=0)

        # build confusion matrix & dump
        confmat = confusion_matrix(targets, predictions)
        tpr_cm[fold-1], tnr_cm[fold-1], f1_cm[fold-1] = calculate_metrics(confmat, n_classes)
        
    tpr_mean = np.mean(tpr_cm, axis=0)
    tpr_std = np.std(tpr_cm, axis=0)
    tpr_var = np.var(tpr_cm, axis=0)
    tnr_mean = np.mean(tnr_cm, axis=0)
    tnr_std = np.std(tnr_cm, axis=0)
    tnr_var = np.var(tnr_cm, axis=0)
    f1_mean = np.mean(f1_cm, axis=0)
    f1_std = np.std(f1_cm, axis=0)
    f1_var = np.var(f1_cm, axis=0)
    print('Config: {} TPR: {} pm {}'.format(config, tpr_mean, tpr_std))
    print('Config: {} TNR: {} pm {}'.format(config, tnr_mean, tnr_std))
    print('Config: {} F1: {} pm {}'.format(config, f1_mean, f1_std))

