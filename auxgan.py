import argparse
import os
import numpy as np
import math
import sys
from scipy import stats, ndimage, special
# from skimage.measure import compare_ssim as ssim
from pytorch_ssim import pytorch_ssim
from sklearn.preprocessing import scale
import random
from ImbalancedSampler import ImbalancedDatasetSampler
# from MapLoader import MapLoader

import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
from MapCrop import MapCrop

# from MapFolder import MapFolder
import torch.nn as nn
import torch.nn.init as weight_init
import torch.nn.functional as F
import torch
from torch.nn import DataParallel
import torch.utils.data.sampler as sampler
from NewMapLoader import MyDataset
from CustomDataset import CustomDataset

try:
    import accimage
except ImportError:
    accimage = None
import pdb
import time
from PIL import Image

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from datetime import timedelta

from setproctitle import setproctitle
setproctitle('Adam W - AUXGAN')

parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=300, help='number of epochs of training')
parser.add_argument('--tolerance', type=int, default=10, help='average number of epochs for training schedule')
parser.add_argument('--threshold', type=float, default=0.01, help='threshold of loss for training schedule')
parser.add_argument('--batch_size', type=int, default=64, help='size of the batches')
parser.add_argument('--test_batch_size', type=int, default=10, help='size of test batches for SSIM')
parser.add_argument('--lrG', type=float, default=0.0002, help='adam: learning rate')
parser.add_argument('--lrD', type=float, default=0.0002, help='SGD: learning rate')
parser.add_argument('--lrDecay', type=float, default=0.10, help='factor by which to decrease SGD learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--latent_dim', type=int, default=100, help='dimensionality of the latent space')
parser.add_argument('--n_classes', type=int, default=4, help='number of classes in dataset')
parser.add_argument('--img_size', type=int, default=64, help='size of each image dimension')
parser.add_argument('--channels', type=int, default=1, help='number of image channels')
parser.add_argument('--sample_interval', type=int, default=400, help='interval between image sampling')
parser.add_argument('--feat_map_dim_D', type=int, default=512, help='highest number of feature maps for Discriminator')
parser.add_argument('--feat_map_dim_G', type=int, default=1024, help='highest number of feature maps for Generator')
parser.add_argument('--dropout', type=float, default=0.25, help='percentage of dropout nodes in training layers')
parser.add_argument('--train_path', type=str, default='/data1/adamw/HDNeuron/GAN_imgs/', help='path to training data')
parser.add_argument('--save_path', type=str, default='/data3/adamw/AUXGAN_thresh/', help='path to save training')
parser.add_argument('--tensor_path', type=str, default='/data1/adamw/HDNeuron/tensors/tensors', help='path to map tensors')
parser.add_argument('--crossfold', type=int, default=5, help='number of cross fold validation runs')
opt = parser.parse_args()
print(opt)

cuda = True if torch.cuda.is_available() else False


class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path


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


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant(m.bias.data, 0.0)


def normalize(img):

    top = img.max()
    bottom = img.min()
    norm = ((img - bottom) / (top - bottom)) * 255

    return norm.round()


def SSIM(real, gen):
    real = real.data[:].cpu().numpy().squeeze()
    gen = gen.data[:].cpu().numpy().squeeze()
    values = np.zeros(len(real)*len(gen))

    # for value in range(len(real)*len(gen)):
    #     iter = 0
    iter = 0
    for i in range(len(gen)):
        img = gen[i]
        img = normalize(img)
        for j in range(len(real)):
            img2 = real[j]
            img2 = normalize(img2)
            values[iter] = pytorch_ssim.ssim(img, img2, data_range=255)
            iter += 1
            # sys.stdout.write('\rcalculating SSIM {}/{}'.format(iter, (len(real) * len(gen))))
    values = np.absolute(values)
    return values.mean()


class Generator(nn.Module):
    def __init__(self, feat_maps):
        super(Generator, self).__init__()

        self.label_emb = nn.Embedding(opt.n_classes, opt.latent_dim)
        self.feat_maps = int(feat_maps)


        self.init_size = opt.img_size // 4  # Initial size before upsampling
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

    def forward(self, noise, labels):
        gen_input = torch.mul(self.label_emb(labels), noise)
        out = self.l1(gen_input)
        out = out.view(out.shape[0], self.feat_maps, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    def __init__(self, feat_maps):

        self.feat_maps = int(feat_maps)

        super(Discriminator, self).__init__()

        def downsampling_block(in_filters, out_filters, bn=True):
            """Returns layers of each discriminator block"""
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1),  # downsampling
                     nn.LeakyReLU(0.2, inplace=True),
                     nn.Dropout2d(opt.dropout)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        # def convolutional_block(in_filters, out_filters, bn=True):
        #     """
        #     :param in_filters: number of input features
        #     :param out_filters: number of output features
        #     :param bn: perform batch normalization
        #     :return: non-downsampled block to increase depth of discriminator
        #     """
        #     block = [nn.Conv2d(in_filters, out_filters, 3, stride=1, padding=1),  # no downsampling
        #              nn.LeakyReLU(0.2, inplace=True),
        #              nn.Dropout2d(0.25)]
        #     if bn:
        #         block.append(nn.BatchNorm2d(out_filters, 0.8))
        #     return block

        self.conv_blocks = nn.Sequential(
            *downsampling_block(opt.channels, self.feat_maps//8, bn=False),
            *downsampling_block(self.feat_maps//8, self.feat_maps//4),
            # *convolutional_block(self.feat_maps//4, self.feat_maps//4),
            *downsampling_block(self.feat_maps//4, self.feat_maps//2),
            # *convolutional_block(self.feat_maps // 2, self.feat_maps // 2),
            *downsampling_block(self.feat_maps//2, self.feat_maps),
        )

        # The height and width of downsampled image
        ds_size = opt.img_size // 2**4

        # Output layers
        self.adv_layer = nn.Sequential(nn.Linear(self.feat_maps*ds_size**2, 1), nn.Sigmoid())
        self.aux_layer = nn.Sequential(nn.Linear(self.feat_maps*ds_size**2, opt.n_classes), nn.Softmax())

    def forward(self, img):
        out = self.conv_blocks(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        label = self.aux_layer(out)

        return validity, label


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


def tensor_loader(filename):
    return np.loadtxt(filename)


def perceptual_loss(dense_fake, debris_fake, spread_fake, diff_fake): # dense_real, debris_real, spread_real, diff_real):

    # load real image dataset, randomly choose 10-100 real image samples (STANDARDIZED)
    real_img_folder = opt.train_path  # '/data1/adamw/HDNeuron/GAN_imgs/'
    dense_imgs = os.listdir('{}/{}/{}'.format(real_img_folder, 'Dense', 'Dense'))
    debris_imgs = os.listdir('{}/{}/{}'.format(real_img_folder, 'Debris', 'Debris'))
    spread_imgs = os.listdir('{}/{}/{}'.format(real_img_folder, 'Spread', 'Spread'))
    diff_imgs = os.listdir('{}/{}/{}'.format(real_img_folder, 'Diff', 'Diff'))

    rand_Dense = random.sample(range(0, len(dense_imgs)), opt.test_batch_size)
    rand_Debris = random.sample(range(0, len(debris_imgs)), opt.test_batch_size)
    rand_Spread = random.sample(range(0, len(spread_imgs)), opt.test_batch_size)
    rand_Diff = random.sample(range(0, len(diff_imgs)), opt.test_batch_size)

    dense_list = list(dense_imgs[i] for i in rand_Dense)
    debris_list = list(debris_imgs[i] for i in rand_Debris)
    spread_list = list(spread_imgs[i] for i in rand_Spread)
    diff_list = list(diff_imgs[i] for i in rand_Diff)

    dense_real = np.zeros((opt.test_batch_size, 1, 64, 64))
    debris_real = np.zeros((opt.test_batch_size, 1, 64, 64))
    spread_real = np.zeros((opt.test_batch_size, 1, 64, 64))
    diff_real = np.zeros((opt.test_batch_size, 1, 64, 64))

    for i in range(opt.test_batch_size):
        dense_real[i, 0, :, :] = random_crop(ndimage.imread(real_img_folder +
                                                            '/Dense/Dense/' + dense_list[i], flatten=True), 64)
        debris_real[i, 0, :, :] = random_crop(ndimage.imread(real_img_folder +
                                                             '/Debris/Debris/' + debris_list[i], flatten=True), 64)
        spread_real[i, 0, :, :] = random_crop(ndimage.imread(real_img_folder +
                                                             '/Spread/Spread/' + spread_list[i], flatten=True), 64)
        diff_real[i, 0, :, :] = random_crop(ndimage.imread(real_img_folder +
                                                           '/Diff/Diff/' + diff_list[i], flatten=True), 64)
    for j in range(opt.test_batch_size):

        debris_fake[j] = normalize_img(debris_fake[j])
        dense_fake[j] = normalize_img(dense_fake[j])
        spread_fake[j] = normalize_img(spread_fake[j])
        diff_fake[j] = normalize_img(diff_fake[j])

    dense_fake = Variable(torch.from_numpy(dense_fake).type(FloatTensor), requires_grad=True)
    debris_fake = Variable(torch.from_numpy(debris_fake).type(FloatTensor), requires_grad=True)
    spread_fake = Variable(torch.from_numpy(spread_fake).type(FloatTensor), requires_grad=True)
    diff_fake = Variable(torch.from_numpy(diff_fake).type(FloatTensor), requires_grad=True)

    dense_real = Variable(torch.from_numpy(dense_real).type(FloatTensor), requires_grad=True)
    debris_real = Variable(torch.from_numpy(debris_real).type(FloatTensor), requires_grad=True)
    spread_real = Variable(torch.from_numpy(spread_real).type(FloatTensor), requires_grad=True)
    diff_real = Variable(torch.from_numpy(diff_real).type(FloatTensor), requires_grad=True)

    debris_metric = pytorch_ssim.ssim(debris_fake, debris_real).abs_()
    dense_metric = pytorch_ssim.ssim(dense_fake, dense_real).abs_()
    spread_metric = pytorch_ssim.ssim(spread_fake, spread_real).abs_()
    diff_metric = pytorch_ssim.ssim(diff_fake, diff_real).abs_()

    return 1 - (0.25 * (debris_metric + dense_metric + spread_metric + diff_metric))


"""
Main - This program trains an auxiliary GAN classifier to produce cell 
colony image patches based on a learned distrubution of the input data.

AuxGAN is trained on a variable loss function that automatically shifts
the weights of three loss parameters (Auxiliary, adversarial, perceptual)
based on learning (either proportional to the SSIM loss or based on the
average improvement of G and D loss). 

Training is performed using an 80/10/10 (train/test/validate) split to keep 
any testing data out of the learning process. 
"""

train_path = opt.train_path
tensor_path = opt.tensor_path
save_path = opt.save_path

feat_dim_D = opt.feat_map_dim_D
feat_dim_G = opt.feat_map_dim_G

randseeds = [6401, 51, 200589, 50098]  # 1046,
thresholds = [0.25, 0.50]

# pdb.set_trace()
for thresh in thresholds:

    for fold, seed in enumerate(randseeds):
        fold += 1
        np.random.seed(seed)

        # TODO: get image filenames, class, or indices and pass to MapCrop

        # Configure image data loader
        transform = transforms.Compose([SmallScale(64),
                                        # transforms.RandomCrop(64),
                                        # MapCrop(64, thing, thresh=0.25),
                                        transforms.Grayscale(),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.RandomVerticalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                        ])

        
        custom_data = CustomDataset(train_path, thresh=thresh)  # , tensor_path)
        
        dataloader = torch.utils.data.DataLoader(custom_data,
                                                 batch_size=opt.batch_size,
                                                 sampler=ImbalancedDatasetSampler(custom_data),
                                                 drop_last=True)


        adversarial_loss = torch.nn.BCELoss()
        auxiliary_loss = torch.nn.CrossEntropyLoss()

        # Initialize generator and discriminator
        generator = DataParallel(Generator(feat_dim_G))
        discriminator = Discriminator(feat_dim_D)

        if cuda:
            generator.cuda()
            discriminator.cuda()
            adversarial_loss.cuda()
            auxiliary_loss.cuda()

        # Initialize weights
        generator.apply(weights_init_normal)
        discriminator.apply(weights_init_normal)

        # Optimizers - Adam works best in both G and D

        optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lrD, betas=(opt.b1, opt.b2))

        Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

        """
        ----------
         Training
        ----------
        """

        G_loss_class = []
        D_loss_class = []
        D_acc_class_real = []
        D_acc_class_fake = []
        SSIM_loss_class = []

        FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

        # assign name to dir for date of training
        new_dir = '{}/{}_{}_19_t{}_{}'.format(save_path, time.localtime().tm_mon,
                                              time.localtime().tm_mday, int(thresh*100), fold + 1)

        # create new directories within date dir
        new_img_dir = '{}/sample_imgs'.format(new_dir)
        os.makedirs(new_img_dir, exist_ok=True)

        new_graph_dir = '{}/performance_graphs'.format(new_dir)
        os.makedirs(new_graph_dir, exist_ok=True)

        new_model_dir = '{}/trained_models'.format(new_dir)
        os.makedirs(new_model_dir, exist_ok=True)

        for epoch in range(opt.n_epochs):

            d_running_loss = g_running_loss = d_running_acc_real = d_running_acc_fake = ssim_running_loss = 0

            for i, (imgs, labels) in enumerate(dataloader):
                
                if epoch == 0:

                    real_grid = make_grid(imgs[:25], nrow=5, normalize=True)
                    save_image(real_grid, '{}/real_images.jpg'.format(new_img_dir))

                    batch_size = imgs.shape[0]

                """
                -----------------
                 Train Generator
                -----------------
                
                determine weighted variable loss parameters
                Loss measures generator's ability to fool the discriminator
                perceptual loss measures generators structural similarity to discriminator
                
                Three loss parameters: 1. Adversarial
                                       2. Auxiliary
                                       3. Perceptual (SSIM)
                
                Experiments: 1. (adaptive) change loss parameter based on ratio of Aux:Adv:SSIM losses
                             2. (adaptive) weights inversely proportional to SSIM loss
                                  a. ex. SSIM: 0.15 --> w_ssim = (0.85) --> w_adv, w_aux = 1-w_ssim = 0.15
                                  b. ex. SSIM: 0.01 --> w_ssim = (0.99)
                             3. (tolerance) choose loss function weight based on change in loss
                                  a. start with adversarial, then auxiliary, then perceptual
                                  b. with average of 5 epochs, tolerance of 0.1
                """

                # Adversarial ground truths
                valid = Variable(FloatTensor(opt.batch_size, 1).fill_(1.0), requires_grad=False)
                fake = Variable(FloatTensor(opt.batch_size, 1).fill_(0.0), requires_grad=False)

                # Configure input
                real_imgs = Variable(imgs.type(FloatTensor))
                labels = Variable(labels.type(LongTensor))

                optimizer_G.zero_grad()

                # Sample noise and labels as generator input
                z = Variable(FloatTensor(np.random.normal(0, 1, (opt.batch_size, opt.latent_dim))))
                test_z = Variable(FloatTensor(np.random.normal(0, 1, (opt.test_batch_size, opt.latent_dim))))
                gen_labels = Variable(LongTensor(np.random.randint(0, opt.n_classes, opt.batch_size)))

                # Generate a batch of images and pass to discriminator
                generator.train()
                discriminator.train()
                gen_imgs = generator(z, gen_labels)
                validity, pred_label = discriminator(gen_imgs)

                losses = adversarial_loss(validity, valid), auxiliary_loss(pred_label, gen_labels) #, ssim_loss
                
                '''
                Experimental Control
                '''
                g_loss = (adversarial_loss(validity, valid) + auxiliary_loss(pred_label, gen_labels)) / 2
                g_running_loss += g_loss.data[0]

                # update loss and take an optimizer step
                g_loss.backward()
                optimizer_G.step()

                """
                ---------------------
                 Train Discriminator
                ---------------------
                """

                optimizer_D.zero_grad()

                # Loss for real images
                real_pred, real_aux = discriminator(real_imgs)
                d_real_loss = (adversarial_loss(real_pred, valid) + auxiliary_loss(real_aux, labels)) / 2

                # Loss for fake images
                fake_pred, fake_aux = discriminator(gen_imgs.detach())
                d_fake_loss = (adversarial_loss(fake_pred, fake) + auxiliary_loss(fake_aux, gen_labels)) / 2

                # Total discriminator loss
                d_loss = (d_real_loss + d_fake_loss) / 2
                d_running_loss += d_loss.data[0]

                # Calculate discriminator accuracy
                # use validation dataset
                pred_real = real_aux.data.cpu().numpy()
                pred_fake = fake_aux.data.cpu().numpy()
                gt_real = labels.data.cpu().numpy()
                gt_fake = gen_labels.data.cpu().numpy()
                d_acc_real = np.mean(np.argmax(pred_real, axis=1) == gt_real)
                d_acc_fake = np.mean(np.argmax(pred_fake, axis=1) == gt_fake)
                d_running_acc_real += d_acc_real
                d_running_acc_fake += d_acc_fake

                d_loss.backward()
                optimizer_D.step()

                sys.stdout.write("\r[Fold {}/{}] [Epoch {}/{}] [Batch {}/{}] [D loss: {:.4f}, acc real/fake: {:.4f}/{:.4f}%] [G loss: {:.4f}] [Train time: {}]"
                                 .format(fold + 1, opt.crossfold,
                                         epoch + 1, opt.n_epochs,
                                         i + 1, len(dataloader),
                                         d_loss.data[0],
                                         100 * d_acc_real, 100 * d_acc_fake,
                                         g_loss.data[0], str(timedelta(seconds=time.time() - start_time))
                                         )
                                 )

            

            # save performance graphs
            # track training loss every epoch
            G_loss_class.append(g_running_loss / len(dataloader))
            D_loss_class.append(d_running_loss / len(dataloader))
            line1, = plt.plot(G_loss_class, label='Generator', linestyle='-', color='r')
            line2, = plt.plot(D_loss_class, label='Discriminator', linestyle='-.', color='g')
            plt.legend(handles=[line1, line2])
            plt.title('Avg. Discriminator/Generator Loss')
            plt.ylabel('Binary Cross Entropy Loss')
            plt.xlabel('Epoch')
            plt.savefig('{}/performance.png'.format(new_graph_dir))
            plt.clf()

            D_acc_class_real.append(d_running_acc_real/len(dataloader))
            D_acc_class_fake.append(d_running_acc_fake/len(dataloader))
            max_acc_real = max(D_acc_class_real)
            max_idx_real = D_acc_class_real.index(max(D_acc_class_real))
            max_acc_fake = max(D_acc_class_fake)
            max_idx_fake = D_acc_class_fake.index(max(D_acc_class_fake))
            line1, = plt.plot(D_acc_class_real, label='Acc. Real', linestyle='-', color='b')
            line2, = plt.plot(D_acc_class_fake, label='Acc. Fake', linestyle=':', color='r')
            plt.legend(handles=[line1, line2])
            plt.title('Discriminator Accuracy vs. Training Epoch')
            plt.plot(max_idx_fake, max_acc_fake, color='k', marker='v')
            plt.plot(max_idx_real, max_acc_real, color='k', marker='v')
            plt.text(max_idx_real-1, max_acc_real-0.1, 'max real = {:.2f}'.format(max_acc_real), color='k')
            plt.text(max_idx_fake-1, max_acc_fake-0.1, 'max fake = {:.2f}'.format(max_acc_fake), color='k')
            plt.ylabel('Overall Class Accuracy')
            plt.xlabel('Epoch')
            plt.savefig('{}/Discriminator_accuracy.png'.format(new_graph_dir))
            plt.clf()

            # save sample images and model snapshot every N epochs
            if epoch % 50 == 0 or epoch == opt.n_epochs - 1:

                train_time = time.time() - start_time
                np.savetxt('{}/train_time.txt'.format(new_dir), np.array([train_time]))

                n_row = 10
                z = Variable(FloatTensor(np.random.normal(0, 1, (n_row ** 2, opt.latent_dim))))
                # Get labels ranging from 0 to n_classes for n rows
                labels = stats.randint(0, 4).rvs(100)  # np.array([num for _ in range(opt.n_classes) for num in range(opt.n_classes)])
                labels = Variable(LongTensor(labels))
                generator.eval()
                gen_imgs = generator(z, labels)
                generator.train()
                sample_grid = make_grid(gen_imgs.data[:], nrow=n_row, normalize=True)
                save_image(sample_grid, '{}/generated_images_epoch_{}.jpg'.format(new_img_dir,
                                                                                  epoch)
                           )
            if (epoch % 100 == 0 and epoch >= 0) or epoch == opt.n_epochs - 1:

                torch.save(discriminator.state_dict(), '{}/discriminator_epoch_{}.ph'.format(new_model_dir,
                                                                                             epoch)
                           )

                torch.save(generator.state_dict(), '{}/generator_epoch_{}.ph'.format(new_model_dir,
                                                                                     epoch)
                           )
