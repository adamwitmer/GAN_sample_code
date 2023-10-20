import numpy as np
import torch
import torch.nn as nn
from torch.nn import DataParallel
from torch.autograd import Variable, grad
from torchvision.utils import save_image, make_grid
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from CustomDataset import CustomDataset_dcgan
from torchvision import datasets
import skimage
import scipy
import os
import scipy.ndimage as ndimage
import argparse
import sys
import pdb
import matplotlib
from ShallowNet import ShallowNet
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
parser.add_argument('--save_path', type=str, default='/data3/adamw/dcgan_w')
parser.add_argument('--crossfold', type=int, default=5, help='number of cross-fold validation')
parser.add_argument('--cuda', type=bool, default=True)
parser.add_argument('--w_lambda', type=int, default=10, help='lambda value for weight penalty')
parser.add_argument('--save_folder', type=str, default='/data1/adamw/entropy_gan')
parser.add_argument('--n_classes', type=int, default=4)
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
class Generator(nn.Module):

    def __init__(self, feat_maps):
        super(Generator, self).__init__()

        self.feat_maps = int(feat_maps)
        self.init_size = opt.img_size // 4
        self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, self.feat_maps*self.init_size**2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(self.feat_maps),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(self.feat_maps, self.feat_maps, 5, stride=1, padding=2),
            nn.BatchNorm2d(self.feat_maps, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(self.feat_maps, self.feat_maps//2, 5, stride=1, padding=2),
            nn.BatchNorm2d(self.feat_maps//2, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.feat_maps//2, opt.channels, 5, stride=1, padding=2),
            nn.Tanh()
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], self.feat_maps, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


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


def normalize_img(img):

    top = img.max()
    bottom = img.min()
    norm = ((img-bottom)/(top-bottom)) * 255.0

    return norm.round()

def sharpen_img(img):

    blurred_img = ndimage.gaussian_filter(img, 1)

    filter_blurred_img = ndimage.gaussian_filter(blurred_img, 1)

    alpha = 2
    sharpened = blurred_img + alpha * (blurred_img - filter_blurred_img)
    return sharpened

def med_filt(img):

    return ndimage.median_filter(img, 3)


"""
I. perform image quality control on generated images by thresholding images based on calculated image entropy 
    A. build histogram of entropy values from generated image patches by class
    B. find optimal threshold for filtering out bad images 
    C. save images representative of various entropy values for observation

II. fashion pre-made datasets containing fake images chosen based on the mean and std. of the entropy distribution
    of the real dataset
    A. build histogram of entropy values from REAL image patches by class
        1. find mean and standard deviation of image entropy values for the entire dataset
    B. filter generated images based on mean and std. of real image entropy distribution
        1. compare histograms of REAL and GENERATED entropy distributions
    C. save dataset with added images, use to train neural network for classification
"""

# paths for trained dcgan models
# diff_gan_path = ''
# debris_gan_path = ''
# # debris_gan_path = '' --> using KLD
# dense_gan_path = ''
# spread_gan_path = ''

# gan_path = ''

Tensor = torch.cuda.FloatTensor


folders = ['high_entropy']  
add_totals = [0, 1000, 2000, 5000]  
configs = ['balanced', '1k', '2k', '5k']  
classes = ['Debris', 'Dense', 'Diff', 'Spread']  #


# initialize proportion of images from each entropy bin per class
# bins --> 5-6, 6-7, >7
proportions = [[0.2749, 0.6855, 0.0396],  # Debris
               [0.2506, 0.6549, 0.0945],  # Dense
               [0.8919, 0.1081, 0.0],     # Diff
               [0.6419, 0.3537, 0.0]],    # Spread

"""load screening CNN to filter out debris images from gan dataset"""
cnn_path = ''
model_path = os.path.join(cnn_path, 'trained_nn.pth')
model = torch.load(model_path)
model = model.cuda()
model.eval()

base_path = ''
opt.save_folder = base_path

for fold, folder in enumerate(folders):

    for cross_fold in range(1):

        for config_num, config in enumerate(configs):

            data_path = '{}/{}_{}'.format(base_path, folder, config)

            class_nums = np.zeros(4)
            for i, clss in enumerate(classes):
                class_nums[i] = len(os.listdir(os.path.join(data_path, 'train', clss)))

            add_num = (class_nums.max() - class_nums) + add_totals[config_num]

            for thing, morph in enumerate(classes):

                entropy_prop = proportions[0][thing]

                new_gan_path = os.path.join(gan_path, morph, 'generator.ph')
                
                data_save = os.path.join(data_path, 'train', morph)
            
                # load trained generator for specific class
                generator = Generator(opt.feat_map_dim_G).cuda()
                generator = DataParallel(generator)
                generator.load_state_dict(torch.load(new_gan_path))
                generator.eval()

                add_imgs = add_num[thing]
                # calculate number of images per bin from multiplying the
                # proportion of entropy images by the number of total images
                bin_numbers = [i * add_imgs for i in entropy_prop]
                bin_numbers = np.floor(bin_numbers).astype(int)
                bin_total = sum(bin_numbers)
                n_add_imgs = 0
                start_position = 0

                while n_add_imgs < int(add_imgs):

                    sys.stdout.write('\r {}_{}, Added Images, {}, {}/{}'
                                     .format(folder, config, morph, n_add_imgs, int(add_imgs)))

                    """dcgan images"""
                    latent_z = Variable(Tensor(np.random.normal(0, 1, (1, opt.latent_dim))))
                    out = generator(latent_z)

                    """auxgan images"""
                    # test_z = Variable(Tensor(np.random.normal(0, 1, (1, opt.latent_dim))))
                    # gen_labels = Variable(torch.cuda.LongTensor([thing]))  # np.random.randint(0, opt.n_classes, opt.batch_size)))
                    # out = generator(test_z, gen_labels)

                    """use ShallowNet to filter Images """
                    if morph != 'Diff':
                        if morph != 'Debris':
                            pred = model.forward(out)
                            pred = nn.functional.softmax(pred).data.cpu().numpy()  # softmax output values
                            pred = np.argmax(pred, axis=1)
                            if pred == 0:
                                continue
                    img = out.cpu().data.numpy().squeeze().squeeze()

                    img = normalize_img(img).astype('uint8')
                    # calculate histogram counts
                    hist, _ = np.histogram(img, bins=256)
                    # remove zero entries from histogram
                    hist = hist[hist != 0]

                    # normalize histogram to probablity distribution (sum of 1)
                    # aka percentage of total values in the array
                    hist = hist / img.size

                    E = -np.sum(hist * np.log2(hist))
                
                    if morph == 'Diff':
                        folder_path = '{}/fake_img_{}.jpg'.format(data_save, n_add_imgs)
                        scipy.misc.imsave(folder_path, img)
                        n_add_imgs += 1
                    else:
                        if E >= 7:
                            folder_path = '{}/fake_img_{}.jpg'.format(data_save, n_add_imgs)
                            scipy.misc.imsave(folder_path, img)
                            n_add_imgs += 1

