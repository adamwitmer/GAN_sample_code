import numpy as np
import torch
import torch.nn as nn
from torch.nn import DataParallel
from torch.autograd import Variable, grad
from torchvision.utils import save_image, make_grid
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from CustomDataset import CustomDataset_dcgan
from pytorch_ssim import pytorch_ssim
from torchvision import datasets
import skimage
import scipy
import os
import argparse
import sys
import pdb
import matplotlib
from matplotlib import rcParams, use
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
parser.add_argument('--save_path', type=str, default='/data3/adamw/AUXGAN_save')
parser.add_argument('--crossfold', type=int, default=5, help='number of cross-fold validation')
parser.add_argument('--cuda', type=bool, default=True)
parser.add_argument('--w_lambda', type=int, default=10, help='lambda value for weight penalty')
parser.add_argument('--save_folder', type=str, default='/data3/adamw/AUXGAN_save')
opt = parser.parse_args()
# print(opt)

rcParams['mathtext.fontset'] = 'stix'
rcParams['font.family'] = 'STIXGeneral'
rcParams['xtick.labelsize'] = 12
rcParams['ytick.labelsize'] = 12
rcParams['axes.labelsize'] = 14
rcParams['axes.titlesize'] = 14
rcParams['legend.fontsize'] = 12

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

class Generator(nn.Module):

    def __init__(self, feat_maps):
        super(Generator, self).__init__()

        self.feat_maps = int(feat_maps)
        self.init_size = opt.img_size // 4
        self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, self.feat_maps*self.init_size**2))

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
            nn.Conv2d(self.feat_maps//2, opt.channels, 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], self.feat_maps, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


def normalize_img(img):

    top = img.max()
    bottom = img.min()
    norm = ((img-bottom)/(top-bottom)) * 255.0

    return norm.round()

"""
perform image quality control on generated images by thresholding images based on calculated image entropy 

1. build histogram of entropy values from generated image patches by class
2. find optimal threshold for filtering out bad images 
3. save images representative of various entropy values for observation
"""

standard_path = '/data3/adamw/dcgan_w/dcgan_w_10_16_19_1'
KLD_path = '/data3/adamw/dcgan_w/dcgan_w_KLD_11_20_19_1'
MSE_path = '/data3/adamw/dcgan_w/dcgan_w_MSE_11_21_19_1'
classes = ['Debris', 'Dense', 'Diff', 'Spread']
loss_params = ['standard']  # , 'MSE']  #  'KLD',
# loss_params = []
colors = ['b', 'r', 'g', 'm']  #
Tensor = torch.cuda.FloatTensor

MSE = nn.MSELoss().cuda()

for param_num, param in enumerate(loss_params):

    for label, morph in enumerate(classes):

        overlap_values = []
        for trial in range(5):

            new_save_folder = os.path.join(opt.save_folder, param, morph)
            os.makedirs(new_save_folder, exist_ok=True)

            # Load real data for histogram comparison
            train_path = os.path.join(opt.train_path, morph)
            transform = transforms.Compose([SmallScale(64),
                                            transforms.RandomCrop(64),
                                            transforms.Grayscale(),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.RandomVerticalFlip(),
                                            transforms.ToTensor(),
                                            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                            ])

            # custom_data = CustomDataset_dcgan(train_path, thresh=0.5)  # , tensor_path)

            train_data = datasets.ImageFolder(train_path,
                                              transform=transform,
                                              # loader=lambda x: Image.open(x).convert('L')
                                              )

            dataloader = torch.utils.data.DataLoader(train_data,
                                                     batch_size=1,
                                                     shuffle=True,
                                                     drop_last=False)
            entropy_values = []
            entropy_values_real = []

            class_num = label
            if param == 'KLD':
                gan_path = os.path.join(KLD_path, morph, 'trained_models/generator.ph')
            elif param == 'MSE':
                gan_path = os.path.join(MSE_path, morph, 'trained_models/generator.ph')
            elif param == 'standard':
                gan_path = os.path.join(standard_path, morph, 'trained_models/generator.ph')

            # load trained generator for specific class
            generator = Generator(opt.feat_map_dim_G).cuda()
            generator.load_state_dict(torch.load(gan_path))
            generator.eval()

            low_end = []
            mid_end = []
            high_end = []

            low_end_real = []
            mid_end_real = []
            high_end_real = []

            n_imgs = 50000
            iter = 0

            # collect entropy values for real images to be plotted
            # determine proportions of data that are within 4 bins (0-5, 5-6, 6-7, >7)
            # calculate histogram
            while iter < n_imgs:

                for thing, (img, _) in enumerate(dataloader):
                    sys.stdout.write('\r Trial: {}, Param: {}, Real {} Image {}/{}'.format(trial+1, param, morph, iter + 1, n_imgs))
                    img = img.numpy().squeeze().squeeze()
                    img = normalize_img(img).astype('uint8')
                    hist, _ = np.histogram(img, bins=256)
                    hist = hist[hist != 0]

                    hist = hist/img.size

                    E = -np.sum(hist * np.log2(hist))

                    # index image entropy value
                    entropy_values_real.append(E)
                    iter += 1
                    if iter == n_imgs:
                        break

            for i in range(n_imgs):

                sys.stdout.write('\rTrial: {}, Param: {}, Generated {} Image {}/{}'.format(trial+1, param, morph, i+1, n_imgs))
                # generate image patches for individual classes using trained dcgan
                latent_z = Variable(Tensor(np.random.normal(0, 1, (1, opt.latent_dim))))
                out = generator(latent_z).cpu().data.numpy().squeeze().squeeze() # dtype='uint8')

                # calculate entropy value from normalized histogram of input image
                # make sure that image is type 'uint8', normalize image
                out = normalize_img(out).astype('uint8')
                # calculate histogram counts
                hist, _ = np.histogram(out, bins=256)
                # remove zero entries from histogram
                hist = hist[hist != 0]

                # normalize histogram to probablity distribution (sum of 1)
                # aka percentage of total values in the array
                hist = hist/out.size

                E = -np.sum(hist * np.log2(hist))
                entropy_values.append(E)
            
            # show histogram of entropy values for generated images
            histogram_values = plt.hist([entropy_values, entropy_values_real[:len(entropy_values)]], bins=100,
                                        label=['Generated', 'Real'], color=[colors[label], 'orange'])
            fake_values = histogram_values[0][0].astype(int)
            real_values = histogram_values[0][1].astype(int)
            overlap = np.sum(np.minimum(fake_values, real_values))/n_imgs
            overlap_values.append(overlap)
            plt.ylim(ymin=0, ymax=3000)  # round(max_value, -3)+1000)
            plt.legend()
            plt.title('Entropy Values for Real Generated {} Images'.format(morph))
            plt.ylabel('Number of Images')
            plt.xlabel('Entropy Values (100 bins) -  Percent Overlap: {:.4f}'.format(overlap))
            if trial == 0:
                plt.savefig('{}/entropy_histogram_good_{}.png'.format(new_save_folder, morph))
            plt.clf()
            
        print('{}: {}({})%'.format(morph, np.mean(overlap_values), np.std(overlap_values)))