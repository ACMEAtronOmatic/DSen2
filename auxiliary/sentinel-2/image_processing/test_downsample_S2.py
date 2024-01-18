#!/usr/bin/env python3

import os ; import sys
from pathlib import Path
import cv2
import rasterio
from scipy import ndimage
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from skimage.measure import block_reduce
from PIL import Image

def denormalize(data):
    return np.uint8(data * 255)
def alpha_correction(band, alpha=0.13, beta = 0.0):
    return np.clip(alpha*band + beta, 0, 255)

def gamma_correction(band, gamma = 2.2):
    return np.power(band, 1.0/gamma)

def normalize_band(band):
    band_min, band_max = (np.amin(data), np.amax(data))
    return ( (band - band_min) / ( (band_max - band_min) ) )

def bright_and_norm(band, alpha=0.13, beta = 0.0, gamma = 2.2, brighten='gamma'):
    if brighten.lower() == 'gamma':
        bright = gamma_correction(band, 1.0/gamma)
    elif brighten.lower() == 'alpha':
        bright = alpha_correction(band, alpha = alpha, beta = beta)
    else:
        bright = band
    band_min, band_max = (np.amin(bright), np.amax(bright))
    band_095 = np.percentile(bright, 99.95)
    return np.clip( (bright - band_min) / ( (band_095 - band_min) ), 0, 1 )

EXTERNALDIR = Path('/run/media/dryglicki/External_01/sentinel-2')
OUTPUTDIR = Path('/home/dryglicki/code/get_sentinel-2_data/images')

B02 = EXTERNALDIR / 'NM-CO/S2A_MSIL2A_20200930T174141_N0214_R098_T13SEB_20200930T221755.SAFE/GRANULE/L2A_T13SEB_A027551_20200930T174219/IMG_DATA/R20m/T13SEB_20200930T174141_B02_20m.jp2'
B03 = EXTERNALDIR / 'NM-CO/S2A_MSIL2A_20200930T174141_N0214_R098_T13SEB_20200930T221755.SAFE/GRANULE/L2A_T13SEB_A027551_20200930T174219/IMG_DATA/R20m/T13SEB_20200930T174141_B03_20m.jp2'
B04 = EXTERNALDIR / 'NM-CO/S2A_MSIL2A_20200930T174141_N0214_R098_T13SEB_20200930T221755.SAFE/GRANULE/L2A_T13SEB_A027551_20200930T174219/IMG_DATA/R20m/T13SEB_20200930T174141_B04_20m.jp2'

with rasterio.open(B02) as blueFile, rasterio.open(B03) as greenFile, \
     rasterio.open(B04) as redFile:
    xvals = np.arange(0,redFile.width,1)
    yvals = np.arange(0,redFile.height,1)
    X1, Y1 = np.meshgrid(xvals,yvals)
    print("Brightening and normalizing.")
    bval = 'gamma'

    blue = bright_and_norm(blueFile.read(1), brighten=bval)
    green = bright_and_norm(greenFile.read(1), brighten=bval)
    red = bright_and_norm(redFile.read(1), brighten=bval)
#   print(np.amax(green))
#   fig = plt.figure()
#   ax = fig.add_subplot()
#   ax.hist(red.flatten(), bins='auto')
#   fig.savefig('histogram_red.png', bbox_inches='tight')
#   exit()

    print("Combining.")
    rgb = np.dstack((red, green, blue))
    print(rgb.shape)
    # Now to try to save the files.

    print("Saving image...")
    im = Image.fromarray(denormalize(rgb), "RGB")
    im.save('pil_test.png', "PNG")
    im.close()
    exit()

    datadown = block_reduce(rgb, block_size=(2,2,1), func=np.mean, cval=np.mean(rgb))
    datadown2 = block_reduce(rgb, block_size=(8,8,1), func=np.mean, cval=np.mean(rgb))

#   vmin = np.amin(data) ; vmax = np.amax(data)
#   datadown = block_reduce(data, block_size=(2,2), func=np.mean, cval=np.mean(data))
#   datadown2 = block_reduce(data, block_size=(16,16), func=np.mean, cval=np.mean(data))
    fig = plt.figure(figsize=(13,5))
    axes = fig.subplots(nrows = 1, ncols = 3)
#   axes[0].imshow(data,cmap='inferno', vmin=vmin, vmax=vmax) ; axes[0].set_title('Original')
#   axes[1].imshow(datadown,cmap='inferno', vmin=vmin, vmax=vmax) ; axes[1].set_title('2x2 down')
#   axes[2].imshow(datadown2,cmap='inferno', vmin=vmin, vmax=vmax) ; axes[2].set_title('16x16 down')
#   fig.savefig('test_downsample.png',dpi=300, bbox_inches='tight')
    print("Creating images.")
    axes[0].imshow(rgb) ; axes[0].set_title('Original')
    axes[1].imshow(datadown) ; axes[1].set_title('2x2 down')
    axes[2].imshow(datadown2) ; axes[2].set_title('8x8 down')
    fig.savefig('test_downsample_rgb.png',dpi=300, bbox_inches='tight')

    # Now to try to save the files.

    im = Image.fromarray(rgb, "RGB")
    im.save('pil_test.png', "PNG")
    im.close()

