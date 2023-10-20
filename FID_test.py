import numpy as np
import torch
import torch.nn as nn
from torch.nn import DataParallel
from torch.autograd import Variable, grad
from torchvision.utils import save_image, make_grid
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
# from CustomDataset import CustomDataset_dcgan
from pytorch_ssim import pytorch_ssim
from torchvision import datasets
# import skimage
import scipy
import os
import argparse
import sys
import pdb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
try:
    import accimage
except ImportError:
    accimage = None

parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=5000, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=64, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--latent_dim', type=int, default=100, help='dimensionality of the latent space')
parser.add_argument('--img_size', type=int, default=64, help='size of each image dimension')
parser.add_argument('--channels', type=int, default=1, help='number of image channels')
parser.add_argument('--sample_interval', type=int, default=400, help='interval between image sampling')
parser.add_argument('--feat_map_dim_G', type=int, default=512, help='highest number of feature maps for G')
parser.add_argument('--feat_map_dim_D', type=int, default=512, help='highest number of feature maps for G and D')
parser.add_argument('--train_path', type=str, default='/data1/adamw/HDNeuron/GAN_imgs')
parser.add_argument('--save_path', type=str, default='/data1/adamw/entropy_gan/fid_datasets')
parser.add_argument('--crossfold', type=int, default=5, help='number of cross-fold validation')
parser.add_argument('--cuda', type=bool, default=True)
parser.add_argument('--w_lambda', type=int, default=10, help='lambda value for weight penalty')
parser.add_argument('--save_folder', type=str, default='/data1/adamw/entropy_gan/fid_datasets')
parser.add_argument('--n_classes', type=int, default=4, help='number of classes in dataset')

opt = parser.parse_args()
# print(opt)


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

"""resgan configuration #1 - single residual layer w/ single convolution """
# class Generator(nn.Module):
#
#     def __init__(self, feat_maps):
#         super(Generator, self).__init__()
#
#         self.feat_maps = int(feat_maps)
#         self.init_size = opt.img_size // 4
#         self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, self.feat_maps*self.init_size**2))
#         self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
#         self.upsample = nn.Upsample(scale_factor=2)
#
#
#         self.conv_1 = nn.Sequential(
#             nn.BatchNorm2d(self.feat_maps),
#             nn.Upsample(scale_factor=2)
#         )
#
#         self.conv_2 = nn.Sequential(
#             nn.Conv2d(self.feat_maps, self.feat_maps, 3, stride=1, padding=1),
#             nn.BatchNorm2d(self.feat_maps, 0.8),
#         )
#
#         self.conv_3 = nn.Sequential(
#             nn.Upsample(scale_factor=2),
#             nn.Conv2d(self.feat_maps, self.feat_maps // 2, 3, stride=1, padding=1),
#             nn.BatchNorm2d(self.feat_maps // 2, 0.8),
#             nn.LeakyReLU(0.2, inplace=True),
#
#         )
#
#         self.conv_4 = nn.Sequential(
#             nn.Conv2d(self.feat_maps // 2, opt.channels, 3, stride=1, padding=1),
#             nn.Tanh()
#         )
#
#
#     def forward(self, z):
#         out = self.l1(z)
#         out = out.view(out.shape[0], self.feat_maps, self.init_size, self.init_size)
#         out = self.conv_1(out)
#         # residual connection 1
#         x1 = out
#         out = self.conv_2(out)
#         out += x1
#         out = self.leaky_relu(out)
#         out = self.conv_3(out)
#         img = self.conv_4(out)
#
#         # img = self.conv_blocks(out)
#         return img

"""resgan config #2 - standard residual block"""
# class Generator(nn.Module):
#
#     def __init__(self, feat_maps):
#         super(Generator, self).__init__()
#
#         self.feat_maps = int(feat_maps)
#         self.init_size = opt.img_size // 4
#         self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, self.feat_maps*self.init_size**2))
#         self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
#         self.upsample = nn.Upsample(scale_factor=2)
#
#
#         self.conv_1 = nn.Sequential(
#             nn.BatchNorm2d(self.feat_maps),
#             nn.Upsample(scale_factor=2)
#         )
#
#         self.conv_2 = nn.Sequential(
#             nn.Conv2d(self.feat_maps, self.feat_maps, 3, stride=1, padding=1),
#             nn.BatchNorm2d(self.feat_maps, 0.8),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(self.feat_maps, self.feat_maps, 3, stride=1, padding=1),
#             nn.BatchNorm2d(self.feat_maps, 0.8),
#         )
#
#         self.conv_3 = nn.Sequential(
#             nn.Upsample(scale_factor=2),
#             nn.Conv2d(self.feat_maps, self.feat_maps // 2, 3, stride=1, padding=1),
#             nn.BatchNorm2d(self.feat_maps // 2, 0.8),
#             nn.LeakyReLU(0.2, inplace=True),
#         )
#
#         self.conv_4 = nn.Sequential(
#             nn.Conv2d(self.feat_maps // 2, opt.channels, 3, stride=1, padding=1),
#             nn.Tanh()
#         )
#
#     def forward(self, z):
#         out = self.l1(z)
#         out = out.view(out.shape[0], self.feat_maps, self.init_size, self.init_size)
#         out = self.conv_1(out)
#         # residual connection 1
#         x1 = out
#         out = self.conv_2(out)
#         out += x1
#         out = self.leaky_relu(out)
#         # upsampling block
#         out = self.conv_3(out)
#         # Tanh block
#         img = self.conv_4(out)
#
#         return img
"""mhgan"""
# class Generator(nn.Module):
#     def __init__(self, ngpu=2, nz=100, ngf=64, nc=1):
#         super(Generator, self).__init__()
#         self.ngpu = ngpu
#         self.main = nn.Sequential(
#             # input is Z, going into a convolution
#             nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
#             nn.BatchNorm2d(ngf * 8),
#             nn.ReLU(True),
#             # state size. (ngf*8) x 4 x 4
#             nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ngf * 4),
#             nn.ReLU(True),
#             # state size. (ngf*4) x 8 x 8
#             nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ngf * 2),
#             nn.ReLU(True),
#             # state size. (ngf*2) x 16 x 16
#             nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ngf),
#             nn.ReLU(True),
#             # state size. (ngf) x 32 x 32
#             nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
#             nn.Tanh()
#             # state size. (nc) x 64 x 64
#         )
#
#     def forward(self, input):
#         if input.is_cuda and self.ngpu > 1:
#             output = nn.parallel.data_parallel(self.main, input,
#                                                range(self.ngpu))
#         else:
#             output = self.main(input)
#         return output


"""aux - mhgan"""
class Generator(nn.Module):
    def __init__(self, ngpu=1, nz=100, ngf=64, nc=1, n_classes=2):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.label_emb = nn.Embedding(n_classes, nz)

        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input, labels):
        if input.is_cuda and self.ngpu > 1:
            output = torch.mul(self.label_emb(labels), input)
            output = self.main(output) # nn.parallel.data_parallel( range(self.ngpu))
        else:
            output = self.main(input)
        return output

"""residual gan config #3 - 6 residual connections"""
# class Generator(nn.Module):
#
#     def __init__(self, feat_maps):
#         super(Generator, self).__init__()
#
#         self.feat_maps = int(feat_maps)
#         self.init_size = opt.img_size // 4
#         self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, self.feat_maps*self.init_size**2))
#         self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
#         self.upsample = nn.Upsample(scale_factor=2)
#
#
#         self.conv_1 = nn.Sequential(
#             nn.BatchNorm2d(self.feat_maps),
#             nn.Upsample(scale_factor=2)
#         )
#
#         self.conv_2 = nn.Sequential(
#             nn.Conv2d(self.feat_maps, self.feat_maps, 3, stride=1, padding=1),
#             nn.BatchNorm2d(self.feat_maps, 0.8),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(self.feat_maps, self.feat_maps, 3, stride=1, padding=1),
#             nn.BatchNorm2d(self.feat_maps, 0.8),
#         )
#
#         self.conv_3 = nn.Sequential(
#             nn.Conv2d(self.feat_maps, self.feat_maps, 3, stride=1, padding=1),
#             nn.BatchNorm2d(self.feat_maps, 0.8),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(self.feat_maps, self.feat_maps, 3, stride=1, padding=1),
#             nn.BatchNorm2d(self.feat_maps, 0.8),
#         )
#
#         self.conv_4 = nn.Sequential(
#             nn.Conv2d(self.feat_maps, self.feat_maps, 3, stride=1, padding=1),
#             nn.BatchNorm2d(self.feat_maps, 0.8),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(self.feat_maps, self.feat_maps, 3, stride=1, padding=1),
#             nn.BatchNorm2d(self.feat_maps, 0.8),
#         )
#
#         self.conv_5 = nn.Sequential(
#             nn.Upsample(scale_factor=2),
#             nn.Conv2d(self.feat_maps, self.feat_maps // 2, 3, stride=1, padding=1),
#             nn.BatchNorm2d(self.feat_maps // 2, 0.8),
#             nn.LeakyReLU(0.2, inplace=True),
#             # nn.Conv2d(self.feat_maps, self.feat_maps, 3, stride=1, padding=1),
#             # nn.BatchNorm2d(self.feat_maps, 0.8),
#             # nn.LeakyReLU(0.2, inplace=True),
#         )
#
#         self.conv_6 = nn.Sequential(
#             nn.Conv2d(self.feat_maps // 2, self.feat_maps // 2, 3, stride=1, padding=1),
#             nn.BatchNorm2d(self.feat_maps // 2, 0.8),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(self.feat_maps // 2, self.feat_maps // 2, 3, stride=1, padding=1),
#             nn.BatchNorm2d(self.feat_maps // 2, 0.8),
#             # nn.LeakyReLU(0.2, inplace=True),
#         )
#
#         self.conv_7 = nn.Sequential(
#             nn.Conv2d(self.feat_maps // 2, self.feat_maps // 2, 3, stride=1, padding=1),
#             nn.BatchNorm2d(self.feat_maps // 2, 0.8),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(self.feat_maps // 2, self.feat_maps // 2, 3, stride=1, padding=1),
#             nn.BatchNorm2d(self.feat_maps // 2, 0.8),
#             # nn.LeakyReLU(0.2, inplace=True),
#         )
#
#         self.conv_8 = nn.Sequential(
#             nn.Conv2d(self.feat_maps // 2, self.feat_maps // 2, 3, stride=1, padding=1),
#             nn.BatchNorm2d(self.feat_maps // 2, 0.8),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(self.feat_maps // 2, self.feat_maps // 2, 3, stride=1, padding=1),
#             nn.BatchNorm2d(self.feat_maps // 2, 0.8),
#             # nn.LeakyReLU(0.2, inplace=True),
#         )
#
#
#         self.conv_9 = nn.Sequential(
#             nn.Conv2d(self.feat_maps // 2, opt.channels, 3, stride=1, padding=1),
#             nn.Tanh()
#         )
#
#     def forward(self, z):
#         out = self.l1(z)
#         out = out.view(out.shape[0], self.feat_maps, self.init_size, self.init_size)
#         out = self.conv_1(out)
#         # residual connection 1
#         x1 = out
#         out = self.conv_2(out)
#         out += x1
#         out = self.leaky_relu(out)
#         # residual connection 2
#         x2 = out
#         out = self.conv_3(out)
#         out += x2
#         out = self.leaky_relu(out)
#         # residual connection 3
#         x3 = out
#         out = self.conv_4(out)
#         out += x3
#         out = self.leaky_relu(out)
#         # upsample layer
#         out = self.conv_5(out)
#         # residual connection 4
#         x4 = out
#         out = self.conv_6(out)
#         out += x4
#         out = self.leaky_relu(out)
#         # residual connection 5
#         x5 = out
#         out = self.conv_7(out)
#         out += x5
#         out = self.leaky_relu(out)
#         # residual connection 6
#         x6 = out
#         out = self.conv_8(out)
#         out += x6
#         out = self.leaky_relu(out)
#         # tanh layer
#         img = self.conv_9(out)
#
#         # img = self.conv_blocks(out)
#         return img


"""auxgan generator"""
# class Generator(nn.Module):
#     def __init__(self, feat_maps):
#         super(Generator, self).__init__()
#
#         self.label_emb = nn.Embedding(opt.n_classes, opt.latent_dim)
#         self.feat_maps = int(feat_maps)
#
#
#         self.init_size = opt.img_size // 4  # Initial size before upsampling
#         self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, self.feat_maps*self.init_size**2))
#
#         self.conv_blocks = nn.Sequential(
#             nn.BatchNorm2d(self.feat_maps),
#             nn.Upsample(scale_factor=2),
#             nn.Conv2d(self.feat_maps, self.feat_maps, 3, stride=1, padding=1),
#             nn.BatchNorm2d(self.feat_maps, 0.8),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Upsample(scale_factor=2),
#             nn.Conv2d(self.feat_maps, self.feat_maps//2, 3, stride=1, padding=1),
#             nn.BatchNorm2d(self.feat_maps//2, 0.8),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(self.feat_maps//2, opt.channels, 3, stride=1, padding=1),
#             nn.Tanh()
#         )
#
#     def forward(self, noise, labels):
#         gen_input = torch.mul(self.label_emb(labels), noise)
#         out = self.l1(gen_input)
#         out = out.view(out.shape[0], self.feat_maps, self.init_size, self.init_size)
#         img = self.conv_blocks(out)
#         return img

"""dcgan generator"""
# class Generator(nn.Module):
#
#     def __init__(self, feat_maps):
#         super(Generator, self).__init__()
#
#         self.feat_maps = int(feat_maps)
#         self.init_size = opt.img_size // 4
#         self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, self.feat_maps*self.init_size**2))
#
#         self.conv_blocks = nn.Sequential(
#             nn.BatchNorm2d(self.feat_maps),
#             nn.Upsample(scale_factor=2),
#             nn.Conv2d(self.feat_maps, self.feat_maps, 3, stride=1, padding=1),
#             nn.BatchNorm2d(self.feat_maps, 0.8),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Upsample(scale_factor=2),
#             nn.Conv2d(self.feat_maps, self.feat_maps//2, 3, stride=1, padding=1),
#             nn.BatchNorm2d(self.feat_maps//2, 0.8),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(self.feat_maps//2, opt.channels, 3, stride=1, padding=1),
#             nn.Tanh()
#         )
#
#     def forward(self, z):
#         out = self.l1(z)
#         out = out.view(out.shape[0], self.feat_maps, self.init_size, self.init_size)
#         img = self.conv_blocks(out)
#         return img

"""wgan generator"""
# class Generator(nn.Module):
#
#     def __init__(self, feat_maps):
#         super(Generator, self).__init__()
#
#         self.feat_maps = int(feat_maps)
#         self.init_size = opt.img_size // 4
#         self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, self.feat_maps*self.init_size**2))
#
#         self.conv_blocks = nn.Sequential(
#             nn.BatchNorm2d(self.feat_maps),
#             nn.Upsample(scale_factor=2),
#             nn.Conv2d(self.feat_maps, self.feat_maps, 5, stride=1, padding=2),
#             nn.BatchNorm2d(self.feat_maps, 0.8),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Upsample(scale_factor=2),
#             nn.Conv2d(self.feat_maps, self.feat_maps//2, 5, stride=1, padding=2),
#             nn.BatchNorm2d(self.feat_maps//2, 0.8),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(self.feat_maps//2, opt.channels, 5, stride=1, padding=2),
#             nn.Tanh()
#         )
#
#     def forward(self, z):
#         out = self.l1(z)
#         out = out.view(out.shape[0], self.feat_maps, self.init_size, self.init_size)
#         img = self.conv_blocks(out)
#         return img


def normalize_img(img):

    top = img.max()
    bottom = img.min()
    norm = ((img-bottom)/(top-bottom))  # * 255.0

    return norm  # .round()

"""
perform image quality control on generated images by thresholding images based on calculated image entropy 

1. build histogram of entropy values from generated image patches by class
2. find optimal threshold for filtering out bad images 
3. save images representative of various entropy values for observation
"""

standard_path = ''
KLD_path = ''
MSE_path = ''
wgan_path = ''
auxgan_path = ''
resgan_path = ''
mhgan_path = ''
mhgan_13_path = ''
classes = ['Diff', 'Spread']  #'Diff',  'Debris',
loss_params = ['mhgan_Diff_Spread']  
colors = ['b', 'r', 'g', 'm']
Tensor = torch.cuda.FloatTensor
FloatTensor = torch.cuda.FloatTensor
LongTensor = torch.cuda.LongTensor

for param_num, param in enumerate(loss_params):

    for morph_class, morph in enumerate(classes):

        new_save_folder = os.path.join(opt.save_folder, param, morph) 
        os.makedirs(new_save_folder, exist_ok=True)

        # Load real data for histogram comparison
        train_path = os.path.join(opt.train_path, morph)
        transform = transforms.Compose([SmallScale(64),
                                        # transforms.RandomCrop(64),
                                        transforms.Grayscale(),
                                        transforms.CenterCrop(64),
                                        # transforms.RandomHorizontalFlip(),
                                        # transforms.RandomVerticalFlip(),
                                        transforms.ToTensor(),
                                        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                        ])

        # custom_data = CustomDataset_dcgan(train_path, thresh=0.5)  # , tensor_path)

        train_data = datasets.ImageFolder(train_path,
                                          transform=transform,
                                          loader=lambda x: Image.open(x).convert('L')
                                          )

        dataloader = torch.utils.data.DataLoader(train_data,
                                                 batch_size=1,
                                                 shuffle=False,
                                                 drop_last=False)

        if param == 'KLD':
            gan_path = os.path.join(KLD_path, morph, 'trained_models/generator.ph')
        elif param == 'MSE':
            gan_path = os.path.join(MSE_path, morph, 'trained_models/generator.ph')
        elif param == 'standard':
            gan_path = os.path.join(standard_path, morph, 'trained_models/generator.ph')
        elif param == 'wgan':
            gan_path = os.path.join(wgan_path, morph, 'generator.ph')
        elif param == 'resgan_3':
            gan_path = os.path.join(resgan_path, morph +'_0', 'trained_models/generator.ph')
        elif param == 'mhgan_3':
            gan_path = '/data2/adamw/mh_gan/save_5/123/netG_epoch_80.pth'
        elif param == 'mhgan':
            if morph == 'Debris':
                gan_path = os.path.join(mhgan_path, morph, 'netG_epoch_110.pth')
            elif morph == 'Dense':
                gan_path = os.path.join(mhgan_path, morph, 'netG_epoch_450.pth')
            elif morph == 'Diff':
                gan_path = os.path.join('/netG_epoch_970.pth')
            elif morph == 'Spread':
                gan_path = os.path.join(mhgan_path, morph, 'netG_epoch_140.pth')
        elif param == 'mhgan_Dense_Spread':
            gan_path = os.path.join(mhgan_13_path, '13', 'netG_epoch_80.pth')
        elif param == 'mhgan_Diff_Spread':
            gan_path = os.path.join(mhgan_13_path, '23', 'netG_epoch_80.pth')
        # gan_path = auxgan_path

        # # load trained generator for specific class
        generator = Generator().cuda()
        generator = DataParallel(generator)
        generator.load_state_dict(torch.load(gan_path))
        generator.eval()

        for i, (_, label) in enumerate(dataloader):


            # if morph == 'Dense':
            #     label = 0
            if morph == 'Diff':
                label = 0
            elif morph == 'Spread':
                label = 1

            gen_labels = torch.Tensor(label).cuda() 
            latent_z = torch.randn(opt.batch_size, 100, 1, 1).cuda()
            out = generator(latent_z, gen_labels)
            out = normalize_img(out)
            sys.stdout.write('\rConfig. {}, Saving {} Image: {}/{}'.format(param, morph, i+1, len(train_data)))
            file_name = os.path.join(new_save_folder, 'fake_{}_img_{}.jpg'.format(morph, i))
            save_image(out.cpu().data[0], file_name, padding=0, normalize=True)

