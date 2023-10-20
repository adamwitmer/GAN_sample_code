"""
This program trains a CNN on image patches from a stem cell microscopy dataset with 4 classes.
CNN's are trained with and without the addition of generated image patches


"""

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
import scipy.misc
import argparse
from CustomDataset import CustomDataset  # , CustomDataset224
from ImbalancedSampler import ImbalancedDatasetSampler
import time
from datetime import timedelta
import pdb
import sys
import os
import cv2
from PIL import Image, ImageFilter
import numpy as np
# from setproctitle import setproctitle
from tessa import dictionary, sfta, chog
import matplotlib
import pickle
from HESCnet import HESCnet
from vgg_19 import Vgg19
import vgg_models
from ShallowNet import ShallowNet
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})
try:
    import accimage
except ImportError:
    accimage = None
# setproctitle('Adam W - cnn')

parser = argparse.ArgumentParser()
parser.add_argument('--d_in', type=int, default=674, help='length of texture feature vector used to train NN')
parser.add_argument('--n_classes', type=int, default=4)
parser.add_argument('--train_path', type=str, default='/data/ucr_data/data1/contrastive_learning/dataset/SimCLR_datasets/four_class/train')  # /data1/adamw/HDNeuron')  # /GAN_imgs')
parser.add_argument('--save_path', type=str, default='/data1/adamw/contrastive_learning/optical_flow/save')  # /data1/adamw/HDNeuron/GAN_imgs_four_classes_128/save/save')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--n_epoch', type=int, default=200, help='number of epochs of training')
parser.add_argument('--crop_thresh', type=float, default=0.5, help='crop threshold for image patches in CustomDataset()')
parser.add_argument('--lr_adam', type=float, default=0.0002, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--lr_sgd', type=float, default=5e-3, help='sgd: learning rate')
parser.add_argument('--momentum_sgd', type=float, default=0.9, help='sgd: momentum')
parser.add_argument('--wd_sgd', type=float, default=0.0001, help='sgd: weight decay - L2 regularization')
parser.add_argument('--save_int', type=int, default=10, help='network save interval')
parser.add_argument('--gan_batches', type=int, default=0, help='number of gan batches to add during training')
opt = parser.parse_args()


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
        if height < self.size or width < self.size:
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


class Generator(nn.Module):
    def __init__(self, feat_maps):
        super(Generator, self).__init__()

        self.label_emb = nn.Embedding(4, 100)
        self.feat_maps = int(feat_maps)


        self.init_size = 64 // 4  # Initial size before upsampling
        self.l1 = nn.Sequential(nn.Linear(100, self.feat_maps*self.init_size**2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(self.feat_maps),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(self.feat_maps, self.feat_maps, 3, stride=1, padding=1),
            nn.BatchNorm2d(self.feat_maps, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(self.feat_maps, self.feat_maps//2, 3, stride=1, padding=1),
            nn.BatchNorm2d(self.feat_maps//2, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.feat_maps//2, 1, 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        gen_input = torch.mul(self.label_emb(labels), noise)
        out = self.l1(gen_input)
        out = out.view(out.shape[0], self.feat_maps, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


def build_dataset(train_path, fld):

    # for testing use first seed, CHANGE FOR CROSS VALIDATION to randseeds[net]
    randseeds = [1046, 6401, 51, 200589, 50098, 24568, 249, 7899, 2, 89000]

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # update customDataset to produce scaled 224 patches
    # custom_data = CustomDataset224(train_path, opt.crop_thresh, normalize=True)
    custom_data = CustomDataset(train_path, opt.crop_thresh, normalize=True)
    weight_data = datasets.ImageFolder(train_path, transform=transforms.Compose([transforms.Resize((224, 224)),
                                                                                 transforms.Grayscale(),
                                                                                 transforms.ToTensor()]
                                                                                )
                                       )

    valid_data = datasets.ImageFolder(train_path, transforms.Compose([SmallScale(64),
                                                                        transforms.Grayscale(),
                                                                        transforms.RandomCrop(64),
                                                                        # transforms.Resize(224),
                                                                        transforms.RandomHorizontalFlip(),
                                                                        transforms.RandomVerticalFlip(),
                                                                        transforms.ToTensor(),
                                                                        normalize]))


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
                                               batch_size=128,
                                               sampler=train_sampler,
                                               shuffle=False,
                                               drop_last=True)

    valid_loader = torch.utils.data.DataLoader(valid_data,
                                               batch_size=128,
                                               sampler=valid_sampler,
                                               shuffle=False,
                                               drop_last=True)

    test_loader = torch.utils.data.DataLoader(custom_data,
                                              batch_size=128,
                                              sampler=test_sampler,
                                              shuffle=False,
                                              drop_last=False)

    weight_loader = torch.utils.data.DataLoader(weight_data,
                                                batch_size=128,
                                                sampler=train_sampler,
                                                shuffle=False)

    return train_loader, valid_loader, weight_loader, test_loader  # , test_idx


def make_gan_imgs(weights, setup):

    # TODO: load trained gan networks
    gan_network= '/data3/adamw/AUXGAN_thresh/8_23_19_t50_5/trained_models/generator_epoch_299.ph'
    generator = DataParallel(Generator(1024).cuda())
    generator.load_state_dict(torch.load(gan_network))
    generator = list(generator.children())
    generator = DataParallel(generator[0])
    generator.eval()

    pdb.set_trace()
    add_imgs = weights.max() - weights
    if setup == 'balance_classes':
        latent_z = Variable(torch.randn(1, 100)).cuda()
        # latent_z = Variable(torch.cuda.FloatTensor(np.random.normal(0, 1, (1, opt.latent_dim)))).cuda()
        labels = Variable(torch.cuda.LongTensor(1).fill_(i))
        out = generator(latent_z, labels)
    # else:
    #     # TODO: balance classes then add number of images implied by setup


def class_weights(data_set):

    class1 = class2 = class3 = class4 = 0
    for point, (x, y) in enumerate(data_set):
        sys.stdout.write('\rGathering class weights...{}/{}'.format(point, len(data_set)))
        class1 += sum(y == 0)
        class2 += sum(y == 1)
        class3 += sum(y == 2)
        class4 += sum(y == 3)
    weights = torch.Tensor([class1, class2, class3, class4])
    return weights.max() / weights


def class_numbers(data_set):

    class1 = class2 = class3 = class4 = 0
    for point, (x, y) in enumerate(data_set):
        sys.stdout.write('\rGathering class numbers...{}/{}'.format(point, len(data_set)))
        class1 += sum(y == 0)
        class2 += sum(y == 1)
        class3 += sum(y == 2)
        class4 += sum(y == 3)
    weights = torch.Tensor([class1, class2, class3, class4])
    return weights


def normalize_vector(vec):

    top = vec.max()
    bottom = vec.min()
    norm = ((vec-bottom)/(top-bottom))

    return norm  # .round()


def normalize_img(img):

    top = img.max()
    bottom = img.min()
    norm = ((img-bottom)/(top-bottom)) * 255

    return norm.round()



data_folders = ['']

n_folds = 1

seeds = [1046, 6401, 51, 200589, 50098] 

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

networks = ['vgg19']  
for folder in data_folders:

    for network in networks:

        for fold in range(n_folds):

            

            train_data = CustomDataset(opt.train_path)

            train_loader = torch.utils.data.DataLoader(train_data,
                                                       batch_size=64,
                                                       sampler=ImbalancedDatasetSampler(train_data),
                                                       shuffle=False)

            # Define model, loss function, optimizer, other parameters
            opt.d_in = 64  # length of input feature vector (HoG --> 2304, multi --> 90, image size --> 64/4096)
            
            if network == 'vgg13':
                model_name = network
                CNN_model = DataParallel(vgg_models.vgg13_bn(num_classes=opt.n_classes).cuda())
            elif network == 'vgg16':
                model_name = network
                CNN_model = DataParallel(vgg_models.vgg16_bn(num_classes=opt.n_classes).cuda())
            elif network == 'vgg19':
                model_name = network
                CNN_model = DataParallel(Vgg19(num_classes=opt.n_classes).cuda())  # vgg_models.vgg19_bn(num_classes=opt.n_classes).cuda())
            elif network == 'ShallowNet':
                model_name = network
                CNN_model = DataParallel(ShallowNet(num_classes=opt.n_classes).cuda())

            """Load Model"""
            print("Model Parameters Reset.")

            """Load pretrained model"""
            
            loss_func = torch.nn.CrossEntropyLoss()  
            optim_func = torch.optim.SGD(CNN_model.parameters(), lr=opt.lr_sgd,
                                         momentum=opt.momentum_sgd, weight_decay=opt.wd_sgd)

            Tensor = torch.cuda.FloatTensor

            # create unique save folder
            save_folder = os.path.join(opt.save_path, '{}_{}_{}'.format(model_name, folder, fold
                                                                                  # time.localtime().tm_mon,
                                                                                  # time.localtime().tm_mday,
                                                                                  # time.localtime().tm_year,
                                                                                 )
                                       )
            os.makedirs(save_folder, exist_ok=True)

            # initialize training monitoring
            train_losses = []
            train_acc = []
            valid_losses = []
            valid_acc = []
            t = time.time()
            for epoch in range(opt.n_epoch):
                if epoch > 0 and epoch % 100 == 0:
                    lr = opt.lr_sgd/10
                    optim_func = torch.optim.SGD(CNN_model.parameters(), lr=lr, momentum=opt.momentum_sgd,
                                                 weight_decay=opt.wd_sgd)
                    print('SGD lr reduced to: {}'. format(lr))

                # initialize batch loss
                train_loss = 0
                total_correct = 0
                count = 0

                # train model
                CNN_model.train()
                for i, (imgs, labels) in enumerate(train_loader):

                    count += len(labels)

                    # train model --> forward pass, compute BCE loss, compute gradient, optimizer step
                    optim_func.zero_grad()
                    output = CNN_model.forward(Variable(imgs.cuda(), requires_grad=False)).cuda()
                    loss = loss_func(output, Variable(labels, requires_grad=False).cuda())
                    loss.backward()
                    optim_func.step()

                    # gather data
                    train_loss += loss.data[0]
                    predictions = np.argmax(output.data.cpu().numpy(), axis=1)
                    correct = (predictions == labels).sum()
                    total_correct += correct

                    # change to output training accuracy and loss parameters, fix 'Train time' output
                    sys.stdout.write(
                        '\rTRAINING {}: Fold: {}, Epoch: {}/{}; Progress: {}%; Train time: {}; Train Loss: {:0.4f}, Train Acc. {:0.4f}%'
                                     .format(folder, fold, epoch + 1, opt.n_epoch, round((i / len(train_loader)) * 100),
                                             str(timedelta(seconds=time.time() - t)), loss.data[0], correct/len(labels)
                                             )
                                     )

                train_losses.append(train_loss/len(train_loader))
                train_acc.append(total_correct/count)

                line1, = plt.plot(train_losses, label='Training Loss', linestyle='-', color='r')
                # line2, = plt.plot(valid_losses, label='Validation Loss', linestyle='-.', color='g')
                plt.legend(handles=[line1])  # , line2])
                plt.title('Training Loss Values')
                plt.ylabel('Binary Cross Entropy Loss')
                plt.xlabel('Epoch')
                plt.savefig('{}/performance.png'.format(save_folder))
                plt.clf()

                line1, = plt.plot(train_acc, label='Train Accuracy', linestyle='-', color='b')
                # line2, = plt.plot(valid_acc, label='Validation Accuracy', linestyle='-.', color='orange')
                plt.legend(handles=[line1])  # , line2])
                plt.title('Training Classification Accuracy')
                plt.ylabel('Classification Accuracy')
                plt.xlabel('Epoch')
                plt.savefig('{}/accuracy.png'.format(save_folder))
                plt.clf()

                if (epoch > 0 and epoch % opt.save_int == 0) or epoch == opt.n_epoch - 1:
                    torch.save(CNN_model, ('{}/trained_nn.pth'.format(save_folder)))

            pickle.dump(train_losses, open(os.path.join(save_folder, 'train_losses.p'), 'wb'))
            pickle.dump(train_acc, open(os.path.join(save_folder, 'train_acc.p'), 'wb'))



