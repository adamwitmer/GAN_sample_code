import os
import cv2
import numpy as np
import skimage
from skimage.filters.rank import entropy
from PIL import Image
import scipy as sp
import matplotlib.pyplot as plt
# from numpy import misc
from scipy.ndimage.morphology import binary_fill_holes
from skimage import data, io, filters, img_as_float
from skimage.filters import threshold_otsu
from skimage import exposure
from skimage.morphology import disk, opening, remove_small_objects
import pdb
import sys


'''
1. rgb to gray 
2. symmetric gaussian filter, size 7
3. entropy filter of 1's, size 3 
4. normalize image (0-1)
    a. img = img - min(min(img))
    b. img = img / max(max(img))
5. morphological opening, disk size 7
6. threshold, > 0.45
7. fill holes
8. remove small objects, < 2000
'''


def normalize_img(img):

    top = img.max()
    bottom = img.min()
    norm = ((img-bottom)/(top-bottom)) * 255

    return norm.round()

classes = ['Debris', 'Dense', 'Diff', 'Spread']
img_folder = '/data1/adamw/HDNeuron/GAN_imgs'
save_folder = '/data1/adamw/HDNeuron/tensors'
example_folder = '/raid/Adam Witmer/AUXGAN/preprocess_imgs'

for i, clss in enumerate(classes):

    tensor_folder = os.path.join(save_folder, 'tensors', clss, clss)
    os.makedirs(tensor_folder, exist_ok=True)

    map_folder = os.path.join(save_folder, 'maps', clss, clss)
    os.makedirs(map_folder, exist_ok=True)

    img_path = os.path.join(img_folder, clss, clss)
    os.chdir(img_path)
    imgs = os.listdir()

    for j, img in enumerate(imgs):

        sys.stdout.write('\rProcessing {} Images: {}/{}'.format(clss, j+1, len(imgs)))

        # load image
        img_gray = cv2.imread(img, 0)
        # cv2.imwrite(os.path.join(example_folder, 'gray_img.png'), img_gray)

        # apply gaussian filter
        img_filt = cv2.GaussianBlur(img_gray, (3, 3), 0)
        # cv2.imwrite(os.path.join(example_folder, 'filtered_img.png'), img_filt)

        # find image entropy
        img_ent = normalize_img(entropy(img_filt, disk(3)))
        # cv2.imwrite(os.path.join(example_folder, 'entropy_img.png'), img_ent)

        # perform morphological opening
        img_open = opening(img_ent, disk(3))
        # cv2.imwrite(os.path.join(example_folder, 'open_img.png'), img_open)

        # binarize image (thresholding)
        thresh = threshold_otsu(img_open)
        _, img_bw = cv2.threshold(img_open, thresh, 255, cv2.THRESH_BINARY)
        # cv2.imwrite(os.path.join(example_folder, 'bw_img.png'), img_bw)

        # hole filling
        img_fill = binary_fill_holes(img_bw)
        # cv2.imwrite(os.path.join(example_folder, 'fill_img.png'), img_fill * 255)

        # remove small objects (<2000)
        img_final = remove_small_objects(img_fill, 2000)
        # cv2.imwrite(os.path.join(example_folder, 'final_img.png'), img_final * 255)

        # save image (as .txt or .png) --> depends on if files are loaded correctly as .png
        img_name = img[:-4]
        np.savetxt('{}/{}.txt'.format(tensor_folder, img_name), img_final.astype(int))
        # cv2.imwrite('{}/{}.jpg'.format(map_folder, img_name), img_final * 255)

        # pdb.set_trace()





