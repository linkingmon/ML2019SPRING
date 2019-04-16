import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy import signal
from scipy import misc
from PIL import Image
import os

# 3*3 Gassian filter
x, y = np.mgrid[-1:2, -1:2]
gaussian_kernel = np.exp(-(x**2+y**2))

# Normalization
gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()

path = 'res50/'
files = []
for r, d, f in os.walk(path):
    for file in f:
        files.append(file)
files.sort()

newdir = 'filter_res50/'
if not os.path.exists(newdir):
    os.makedirs(newdir)

for num, file in enumerate(files):

    img = Image.open(path + file)
    image = np.fromstring(img.tobytes(), dtype=np.uint8)
    image = image.reshape((img.size[1], img.size[0], 3)).astype('int')

    grad0 = signal.convolve2d(image[:,:,0], gaussian_kernel, boundary='symm', mode='same')
    grad1 = signal.convolve2d(image[:,:,1], gaussian_kernel, boundary='symm', mode='same')
    grad2 = signal.convolve2d(image[:,:,2], gaussian_kernel, boundary='symm', mode='same')
    filter_image = np.stack((grad0, grad1, grad2), axis = 2)

    im_filt = Image.fromarray(filter_image.astype(np.uint8))
    im_filt.save(newdir + file)