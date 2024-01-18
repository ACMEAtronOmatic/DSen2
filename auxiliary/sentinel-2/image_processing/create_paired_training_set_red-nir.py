#!/usr/bin/env python3

import os ; import sys
from pathlib import Path
import rasterio
from PIL import Image
from scipy import ndimage
import numpy as np
from skimage.measure import block_reduce
import skimage
import gc
from datetime import datetime, timedelta
from argparse import ArgumentParser
import shutil
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

RESOLUTIONS = [ '60m', '20m', '10m' ]

def normalize_band(band, zero=False):
    band_min, band_max = (np.amin(band), np.amax(band))
    if zero: band_min = 0.0
    return ( (band - band_min) / ( (band_max - band_min) ) )

def denormalize(data):
    return np.uint8(data * 255)

def to_uint8(data):
    return np.uint8(np.clip(data,0.0,255.0))

def normalize_and_clip(data):
    mi, ma = np.percentile(data, (0.5,99.5))
#   mi = np.amin(data)
#   ma = np.amax(data)
    band_data = np.clip(data, mi, ma)
    return (band_data - mi) / (ma - mi)

def main():

    description = 'Build training dataset from Sentinel-2 NIR and SWIR bands.'

    parser = ArgumentParser(description=description)

    parser.add_argument('-r', '--reduction_factor', default = 4, help = 'Reduction factor from source data. Default: 4.')
    parser.add_argument('-d', '--data_directory', default = '/home/dryglicki/data/sentinel-2/source', help = 'Source directory')
    parser.add_argument('-p', '--parent_directory', default = '/home/dryglicki/data/sentinel-2', help = 'Parent directory.')
    parser.add_argument('-l', '--label', default = 'training', help = 'Label for the directory. Should be either training or testing.')
    parser.add_argument('-sub', '--subdomain_size', default = 256, help = 'Subdomain size, value squared. Default: 256.')
    parser.add_argument('--clean', action='store_true', help = 'If directories exist, wipe them.')

    args = parser.parse_args()

    RFACTOR = int(args.reduction_factor)
    EXTERNALDIR = Path(args.parent_directory)
    SUBDOM = int(args.subdomain_size)
    DATADIR = Path(args.data_directory)
    CLEAN = args.clean
    LABEL = args.label.lower()
    if LABEL not in ['training', 'testing']:
        raise Exception("Label must be either training or testing.")

    if SUBDOM % RFACTOR != 0:
        raise Exception("Subdomain size must be divisible by reduction factor.")

    BANDS = ['B04', 'B8A']

    pairedDir = EXTERNALDIR / 'paired'
    trainDir = pairedDir / args.label.lower()
    valDir = pairedDir / 'validation'

    trainRed = trainDir / 'red'
    trainNIRLow = trainDir / 'nirLow'
    trainNIRHi  = trainDir / 'nirHigh'

    valRed    = valDir / 'red'
    valNIRLow = valDir / 'nirLow'
    valNIRHi  = valDir / 'nirHigh'

    for dir_ in [ trainRed, trainNIRLow, trainNIRHi, valRed, valNIRLow, valNIRHi ]:
        if not dir_.exists(): dir_.mkdir(parents=True)

    counter = 0
    for item in DATADIR.glob('*'):
        if not item.is_dir(): continue
        for SafeDir in item.glob('*.SAFE'):
            granuleDir = SafeDir / 'GRANULE'
            for SectorDir in granuleDir.glob('*'):
                ImageDir = SectorDir / 'IMG_DATA'
                MidResDir = ImageDir / f'R{RESOLUTIONS[1]}'
                print("Currently inside: ",SectorDir)
                for file in MidResDir.glob('*jp2'):
                    band = file.name.split('_')[2]
                    passtime = datetime.strptime(file.name.split('_')[1], '%Y%m%dT%H%M%S')
                    if band == BANDS[0]: File01 = file
                    if band == BANDS[1]: File02 = file
                with rasterio.open(File01) as red, rasterio.open(File02) as nir:
                    print(File01)
                    print(File02)
                    w = red.width ; h = red.height
                    dataRedRaw = red.read(1) ; dataNIRRaw = nir.read(1)
                    dataRed = denormalize(normalize_and_clip(dataRedRaw)) ; dataNIR = denormalize(normalize_and_clip(dataNIRRaw))
#                   print(np.amax(dataRed), np.amin(dataRed))
#                   print(np.amax(dataNIR), np.amin(dataNIR))
#                   print(dataNIR.dtype)
                    filtered = skimage.filters.gaussian(np.float32(dataNIR), sigma = 1./np.float32(RFACTOR), mode='reflect')
#                   print(np.amax(filtered), np.amin(filtered))
#                   print(filtered.dtype)
                    datadown = np.uint8(block_reduce(filtered, block_size = (RFACTOR,RFACTOR), func=np.mean, cval=np.mean(filtered)))
#                   print(datadown.dtype)
#                   print(np.amax(datadown), np.amin(datadown))
#                   exit()
#                   fig = plt.figure(figsize=(8,8))
#                   ax = fig.add_subplot()
#                   ax.imshow(datadown,cmap='bone')
#                   fig.savefig('datadown-test.png')
                    LOSUBDOM = SUBDOM // RFACTOR
                    xNumHiRes = w // SUBDOM
                    yNumHiRes = h // SUBDOM
                    xNumLowRes, yNumLowRes = [ i // LOSUBDOM for i in np.shape(datadown) ]

#                   print(SUBDOM, LOSUBDOM)
#                   print(xNumHiRes, xNumLowRes)
#                   exit()

                    xchunks = min(xNumHiRes,xNumLowRes)
                    ychunks = min(xNumHiRes,xNumLowRes)

                    for x in range(xchunks):
                        for y in range(ychunks):

                            lb = x * SUBDOM ; rb = (x+1) * SUBDOM
                            bb = y * SUBDOM ; tb = (y+1) * SUBDOM

                            domsize = (rb-lb) * (tb - bb)

                            checkSize = int(0.15 * domsize)

                            # Red chunks to training
                            Chunk = dataRed[lb:rb, bb:tb]
                            if np.count_nonzero(Chunk == np.uint8(0)) > checkSize: continue # Check for darkness or edge of scan
                            fname = f'{counter:09d}.png'
                            im = Image.fromarray(Chunk, 'L')
                            filePath = trainRed / fname
                            print("Currently writing out to: ",filePath)
                            im.save(filePath, "PNG")
                            im.close()
                            gc.collect()

                            # NIR High chunks to training
                            Chunk = dataNIR[lb:rb, bb:tb]
                            fname = f'{counter:09d}.png'
                            im = Image.fromarray(Chunk, 'L')
                            filePath = trainNIRHi / fname
                            im.save(filePath, "PNG")
                            im.close()
                            gc.collect()

                            # NIR Low chunks to training
                            lb = x * LOSUBDOM ; rb = (x+1) * LOSUBDOM
                            bb = y * LOSUBDOM ; tb = (y+1) * LOSUBDOM

                            Chunk = datadown[lb:rb, bb:tb]
                            im = Image.fromarray(Chunk, 'L')
                            filePath = trainNIRLow / fname
                            im.save(filePath, "PNG")
                            im.close()
                            gc.collect()

                            counter += 1

    if LABEL == 'training':
    # Now to create a validation set.
        rng = np.random.default_rng(seed=42)
        ints = rng.integers(0, counter, size = (counter-1) // 5) # Take 20% of the training sample for validation

        for num in ints:
            fileString = f'{num:09d}.png'
            testPath = trainRed / fileString
            if not testPath.exists(): continue
            for trainPath, valPath in \
                zip( [trainRed, trainNIRHi, trainNIRLow],
                     [valRed, valNIRHi, valNIRLow] ):
                fromPath = trainPath / fileString
                toPath = valPath / fileString
                shutil.move(fromPath, toPath)


if __name__ == "__main__":
    main()
