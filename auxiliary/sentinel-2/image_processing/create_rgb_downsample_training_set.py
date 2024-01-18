#!/usr/bin/env python3

import os ; import sys
from pathlib import Path
import cv2
import rasterio
from PIL import Image
from scipy import ndimage
import numpy as np
from skimage.measure import block_reduce
import gc
from datetime import datetime, timedelta

def alpha_correction(band, alpha=0.13, beta = 0.0):
    return np.clip(alpha*band + beta, 0, 255)

def gamma_correction(band, gamma = 2.2):
    return np.power(band, 1.0/gamma)

def normalize_band(band, zero=False):
    band_min, band_max = (np.amin(band), np.amax(band))
    if zero: band_min = 0.0
    return ( (band - band_min) / ( (band_max - band_min) ) )

def bright_and_norm(band, alpha=0.13, beta = 0.0, gamma = 2.2, brighten='gamma', lowpct=1.0, highpct=99.0):
    if brighten.lower() == 'gamma':
        bright = gamma_correction(band, 1.0/gamma)
    elif brighten.lower() == 'alpha':
        bright = alpha_correction(band, alpha = alpha, beta = beta)
    else:
        bright = band
    band_min, band_max = (np.amin(bright), np.amax(bright))
    band_max_pct = np.percentile(bright, highpct)
    band_min_pct = np.percentile(bright, lowpct)
    return np.clip( (bright - band_min_pct) / ( (band_max_pct - band_min_pct) ), 0, 1 )

def denormalize(data):
    return np.uint8(data * 255.0)

EXTERNALDIR = Path('/run/media/dryglicki/External_01/sentinel-2')
TRAININGDIR = EXTERNALDIR / 'training'
RGBDIR = TRAININGDIR / 'RGB'
NIRDIR = TRAININGDIR / 'NIR3D'
HIRESDIRRGB = RGBDIR / 'hires'
LOWRESDIRRGB = RGBDIR / 'lowres'
HIRESDIRNIR = NIRDIR / 'hires'
LOWRESDIRNIR = NIRDIR / 'lowres'
RESOLUTIONS = [ '60m', '20m', '10m' ]
BVAL = 'gamma'
IMG_MODE = 'RGB'

CHECKFILE = '/home/dryglicki/code/get_sentinel-2_data/images/filelist.txt'

if not HIRESDIRRGB.exists():  HIRESDIRRGB.mkdir(parents=True)
if not LOWRESDIRRGB.exists(): LOWRESDIRRGB.mkdir(parents=True)
if not HIRESDIRNIR.exists():  HIRESDIRNIR.mkdir(parents=True)
if not LOWRESDIRNIR.exists(): LOWRESDIRNIR.mkdir(parents=True)

sizeHiRes  = 256
sizeLowRes = 128

def main():
    sectorIDs = []
    with open(CHECKFILE,'r') as f:
        for line in f.readlines():
            seg = line.split('_')
            sectorIDs.append('_'.join(seg[:4]))
    print(sectorIDs)
    counter = 0
    for item in EXTERNALDIR.glob('*'):
        if not item.is_dir(): continue
        if "training" in item.name: continue
        for SafeDir in item.glob('*.SAFE'):
            granuleDir = SafeDir / 'GRANULE'
            for SectorDir in granuleDir.glob('*'):
                sectorName = SectorDir.name
                if sectorName not in sectorIDs: continue
                ImageDir = SectorDir / 'IMG_DATA'
                MidResDir = ImageDir / f'R{RESOLUTIONS[1]}'
                for file in MidResDir.glob('*jp2'):
                    band = file.name.split('_')[2]
                    passtime = datetime.strptime(file.name.split('_')[1], '%Y%m%dT%H%M%S')
                    if band == 'B02': B02File = file # Blue
                    if band == 'B03': B03File = file # Green
                    if band == 'B04': B04File = file # Red
                    if band == 'B8A': B8AFile = file # NIR
                with rasterio.open(B02File) as blueFile, rasterio.open(B03File) as greenFile, \
                    rasterio.open(B04File) as redFile, rasterio.open(B8AFile) as nirFile:
                    w = redFile.width ; h = redFile.height
                    print("Brightening and normalizing.")
                    blue = bright_and_norm(blueFile.read(1), brighten=BVAL)
                    green = bright_and_norm(greenFile.read(1), brighten=BVAL)
                    red = bright_and_norm(redFile.read(1), brighten=BVAL)
                    nir = normalize_band(nirFile.read(1), zero=True)
                    # Stacking the channels.
                    rgb = np.dstack((red, green, blue))
                    datadown = block_reduce(rgb, block_size=(2,2,1), func=np.mean, cval=np.mean(rgb))
                    nirdown = block_reduce(nir, block_size=(2,2), func=np.mean, cval=np.mean(nir))
                    xNumHiRes = w // sizeHiRes
                    yNumHiRes = h // sizeHiRes
                    xNumLowRes, yNumLowRes, _ = [ i // sizeLowRes for i in np.shape(datadown) ]

                    xchunks = min(xNumHiRes,xNumLowRes)
                    ychunks = min(xNumHiRes,xNumLowRes)

                    for x in range(xchunks):
                        for y in range(ychunks):
                            lb = x * sizeHiRes ; rb = (x+1) * sizeHiRes
                            bb = y * sizeHiRes ; tb = (y+1) * sizeHiRes
                            hiResChunkRGB  = denormalize(rgb[lb:rb, bb:tb, :])
                            hiResChunkNIR  = denormalize(nir[lb:rb, bb:tb])
                            hiResChunkNIRFull = np.repeat(hiResChunkNIR[:,:,None], 3, axis=2)
                            lb = x * sizeLowRes ; rb = (x+1) * sizeLowRes
                            bb = y * sizeLowRes ; tb = (y+1) * sizeLowRes
                            lowResChunkRGB = denormalize(datadown[lb:rb , bb:tb, :])
                            lowResChunkNIR = denormalize(nirdown[lb:rb , bb:tb])
                            lowResChunkNIRFull = np.repeat(lowResChunkNIR[:,:,None], 3, axis=2)
                            if np.count_nonzero(hiResChunkNIRFull == np.uint8(0)) > 0: continue
                            fname = f'{counter:08d}.png'
                            print("Writing file: ",fname)
                            # RGB files first.
                            im = Image.fromarray(hiResChunkRGB, "RGB")
                            filePath = HIRESDIRRGB / fname
                            im.save(filePath, "PNG")
                            im.close()
                            gc.collect()

                            im = Image.fromarray(lowResChunkRGB, "RGB")
                            filePath = LOWRESDIRRGB / fname
                            im.save(filePath, "PNG")
                            im.close()
                            gc.collect()
                            # NIR file.
                            im = Image.fromarray(hiResChunkNIRFull, "RGB")
                            filePath = HIRESDIRNIR / fname
                            im.save(filePath, "PNG")
                            im.close()
                            gc.collect()

                            im = Image.fromarray(lowResChunkNIRFull, "RGB")
                            filePath = LOWRESDIRNIR / fname
                            im.save(filePath, "PNG")
                            im.close()
                            gc.collect()

                            counter += 1
if "__main__" == __name__:
    main()
