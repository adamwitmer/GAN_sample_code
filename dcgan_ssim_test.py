"""

calculate the Structural Similarity Index (SSIM) between generated and real images
from various implementations of Generative Adversarial Networks across classes

Pseudo-code:
1. Load trained generator networks; real image dataset
2. Randomly choose 10 - 100 images from the real image datasets
3. Generate 100 images for each class using the generators
4. Calculate the SSIM between each generated image and 10 - 100 real image samples
    a. average the SSIM for each class across generators
5. DECLARE YOURSELF KING OF EVERYTHING!

"""

import torch
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
from torchsample.callbacks import EarlyStopping
import torchsample.transforms as tstransforms
import numpy as np
from skimage import data, img_as_float
# from skimage.measure import compare_ssim as ssim
import os
import pdb
import random
from PIL import Image
from scipy import ndimage
from sklearn.preprocessing import scale
from torch.autograd import Variable
from pytorch_ssim import pytorch_ssim
import sys
try:
    import accimage
except ImportError:
    accimage = None
import setproctitle

# random.seed(1000)


# def perceptual_loss(fake_imgs, real_imgs):
#
#     fake_fake = Variable(torch.from_numpy(fake_imgs).type(FloatTensor), requires_grad=True)
#     real_real = Variable(torch.from_numpy(real_imgs).type(FloatTensor), requires_grad=True)
#     ssim_metric = pytorch_ssim.ssim(fake_fake, real_real).abs_()
#
#     return 1 - ssim_metric

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


def normalize_img(img):

    top = img.max()
    bottom = img.min()
    norm = ((img-bottom)/(top-bottom)) * 255

    return norm.round()


def random_crop(img, sz):

    w, h = img.shape

    # resize image if too small
    if w < sz or h < sz:
        ratio = min(w/sz, h/sz)
        w /= ratio
        h /= ratio
        w = round(w)
        h = round(h)
        img = np.resize(img, (w, h))

    # crop and return
    x = random.randint(0, w-sz)
    y = random.randint(0, h-sz)

    return img[x:x+sz, y:y+sz]


# for dcGAN
class Generator(nn.Module):
    def __init__(self, feat_maps):
        super(Generator, self).__init__()

        self.feat_maps = int(feat_maps)
        self.init_size = 64 // 4
        self.l1 = nn.Sequential(nn.Linear(100, self.feat_maps * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(self.feat_maps),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(self.feat_maps, self.feat_maps, 3, stride=1, padding=1),
            nn.BatchNorm2d(self.feat_maps, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(self.feat_maps, self.feat_maps // 2, 3, stride=1, padding=1),
            nn.BatchNorm2d(self.feat_maps // 2, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.feat_maps // 2, 1, 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], self.feat_maps, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


# I. Evaluate trained DCGAN models (w/, w/o) perceptual loss func. using SSIM and MSE indices
#       a. based on 5-fold validation
#       b. determine the average SSIM and MSE scores for each class
MSE = nn.MSELoss().cuda()

FloatTensor = torch.cuda.FloatTensor
LongTensor = torch.cuda.LongTensor

classes = ['Dense', 'Debris', 'Diff', 'Spread']

# standard_folders = ['1_2_18_0', '1_2_18_1', '1_3_18_2', '1_3_18_3', '1_3_18_4']
# ssim_folders = ['1_16_18_0', '1_18_18_1', '1_19_18_2', '1_21_18_3', '1_23_18_4']
dcgan_folders = ['5_3_19_ssim_0', '5_5_19_ssim_2', '5_7_19_ssim_3']

mean_array = np.zeros((len(dcgan_folders), 4, 2))  # x folders, 4 classes, 1 metric, 1 std.
std_array = np.zeros((len(dcgan_folders), 4, 2))

randseeds = [1046, 51, 200589] # , seed 4 --> 50098]  # seed 1 --> 6401,
data_folder = '/raid/Adam Witmer/AUXGAN/GAN_imgs'

for fold, seed in enumerate(randseeds):

    # for y, data_class in enumerate(classes):

    folder = dcgan_folders[fold]

    img_data = datasets.ImageFolder(data_folder, transforms.Compose([SmallScale(64),
                                                                     transforms.Grayscale(),
                                                                     transforms.RandomCrop(64),
                                                                     # transforms.RandomHorizontalFlip(),
                                                                     # transforms.RandomVerticalFlip(),
                                                                     transforms.ToTensor()]))

    num_train = len(img_data)
    indices = list(range(num_train))
    split = int(np.floor(0.2 * num_train))

    np.random.seed(seed)
    np.random.shuffle(indices)

    train_idx, test_idx = indices[split:], indices[:split]

    # train_sampler = sampler.SubsetRandomSampler(train_idx)
    # valid_sampler = sampler.SubsetRandomSampler(valid_idx)
    test_sampler = sampler.SubsetRandomSampler(test_idx)

    dataloader = torch.utils.data.DataLoader(img_data,
                                             batch_size=1,
                                             sampler=test_sampler,
                                             shuffle=False,
                                             drop_last=False)

    class1 = class2 = class3 = class4 = 0
    for i, (_, cls) in enumerate(dataloader):
        sys.stdout.write('\rGetting class nubmers: {}/{}'.format(i, len(dataloader)))
        class1 += sum(cls == 0)
        class2 += sum(cls == 1)
        class3 += sum(cls == 2)
        class4 += sum(cls == 3)

    class_nums = np.array([class1, class2, class3, class4])

    # dcGAN 512 feature maps
    dcGAN_debris = '/raid/Adam Witmer/DCGAN_standard/{}/Debris/trained_models'.format(folder)
    dcGAN_dense = '/raid/Adam Witmer/DCGAN_standard/{}/Dense/trained_models'.format(folder)
    dcGAN_spread = '/raid/Adam Witmer/DCGAN_standard/{}/Spread/trained_models'.format(folder)
    dcGAN_diff = '/raid/Adam Witmer/DCGAN_standard/{}/Diff/trained_models'.format(folder)

    dcgan_debris = Generator(512).cuda()
    dcgan_dense = Generator(512).cuda()
    dcgan_spread = Generator(512).cuda()
    dcgan_diff = Generator(512).cuda()

    dcgan_debris.load_state_dict(torch.load(dcGAN_debris + '/generator_epoch_999.ph'))
    dcgan_dense.load_state_dict(torch.load(dcGAN_dense + '/generator_epoch_999.ph'))
    dcgan_spread.load_state_dict(torch.load(dcGAN_spread + '/generator_epoch_999.ph'))
    dcgan_diff.load_state_dict(torch.load(dcGAN_diff + '/generator_epoch_999.ph'))

    dcgan_debris.eval()
    dcgan_dense.eval()
    dcgan_spread.eval()
    dcgan_diff.eval()

    num_times = 1

    # indexing matrices for ssim values, to be divided
    # by total number of instances in classes
    ssims = np.zeros(4)
    mses = np.zeros(4)

    for do in range(num_times):

        for i, (img, label) in enumerate(dataloader):

            sys.stdout.write('\r processing images, fold: {}/{}, round: {}/{}, image: {}/{}'
                             .format(fold+1, len(randseeds), do+1, num_times, i+1, len(dataloader)))

            latent_z = Variable(torch.randn(1, 100)).cuda()
            labels = Variable(label)
            if label.numpy() == 0:
                fake_img = dcgan_debris(latent_z).cpu().data[:].numpy()
            elif label.numpy() == 1:
                fake_img = dcgan_dense(latent_z).cpu().data[:].numpy()
            elif label.numpy() == 3:
                fake_img = dcgan_spread(latent_z).cpu().data[:].numpy()
            elif label.numpy() == 2:
                fake_img = dcgan_diff(latent_z).cpu().data[:].numpy()

            fake_img = normalize_img(fake_img)

            real_img = normalize_img(img.numpy())
            real_data = Variable(torch.from_numpy(real_img).type(FloatTensor), requires_grad=False).cuda()
            fake_data = Variable(torch.from_numpy(fake_img).type(FloatTensor), requires_grad=False).cuda()

            ssims[label] += pytorch_ssim.ssim(fake_data, real_data).abs_().data[0]
            mses[label] += MSE(fake_data, real_data).data[0]

    mean_array[fold, :, 0] = ssims / (class_nums * num_times)
    mean_array[fold, :, 1] = mses / (class_nums * num_times)
pdb.set_trace()

        # # load real image dataset, randomly choose 10-100 real image samples (STANDARDIZED)
        # real_img_folder = '/data1/adamw/HDNeuron/GAN_imgs/'
        # real_imgs = os.listdir('{}/{}/{}'.format(real_img_folder, data_class, data_class))
        #
        # img_metrics = np.zeros((100, 2))
        # count = 0
        #
        # for z in range(100):
        #
        #     sys.stdout.write('\rTesting... Fold: {}, Class: {}, Iter: {}/100'.format(x, data_class, z))
        #     rand_imgs = random.sample(range(0, len(real_imgs)), 100)
        #     img_list = list(real_imgs[i] for i in rand_imgs)
        #     img_array = np.zeros((100, 64, 64))
        #
        #     for i in range(len(rand_imgs)):
        #
        #         img_array[i, :, :] = random_crop(ndimage.imread(real_img_folder +
        #                                                           '/{}/{}/'.format(data_class, data_class) +
        #                                                           img_list[i], flatten=True), 64)
        #
        #
        #     # Generate 100 images for each class using the generators
        #     latent_z = Variable(torch.cuda.FloatTensor(np.random.normal(0, 1, (100, 100)))).cuda()
        #     dcgan_imgs = dcgan(latent_z).cpu().data[:].numpy()
        #
        #     # normalize images between 0-1 and then 0-255 for comparison to real images,
        #     for i in range(100):
        #
        #         dcgan_imgs[i] = normalize_img(dcgan_imgs[i])
        #
        #     # 100 images in each class
        #     real_data = Variable(torch.from_numpy(img_array).type(FloatTensor), requires_grad=False).unsqueeze(1)
        #     fake_data = Variable(torch.from_numpy(dcgan_imgs).type(FloatTensor), requires_grad=False)
        #
        #     # each real/fake comparison produces one aggregate loss variable
        #     # which results in 100 values to be averaged for each class (10000 in total)
        #     img_metrics[z, 0] = pytorch_ssim.ssim(fake_data, real_data).abs_().data[0]
        #     img_metrics[z, 1] = mse_metric = MSE(fake_data, real_data).data[0]
        #
        # mean_array[x, y] = img_metrics.mean(0)
        # std_array[x, y] = img_metrics.std(0)

# pdb.set_trace()

