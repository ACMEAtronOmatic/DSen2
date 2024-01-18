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
from argparse import ArgumentParser

RESOLUTIONS = [ '60m', '20m', '10m' ]

def normalize_band(band, zero=False):
    band_min, band_max = (np.amin(band), np.amax(band))
    if zero: band_min = 0.0
    return ( (band - band_min) / ( (band_max - band_min) ) )

def denormalize(data):
    return np.uint8(data * 255.0)

def main():

    description = 'Build training dataset from Sentinel-2 NIR and SWIR bands.'

    parser = ArgumentParser(description=description)

    parser.add_argument('-r', '--reduction_factor', default = 2, help = 'Reduction factor from source data. Default: 2.')
    parser.add_argument('-d', '--data_directory', default = '/run/media/dryglicki/External_01/sentinel-2/source', help = 'Source directory')
    parser.add_argument('-p', '--parent_directory', default = '/run/media/dryglicki/External_01/sentinel-2', help = 'Parent directory.')
    parser.add_argument('-l', '--label', default = 'training', help = 'Label for the directory. Should be either training or testing.')
    parser.add_argument('-sub', '--subdomain_size', default = 256, help = 'Subdomain size, value squared. Default: 256.')
    parser.add_argument('-c', '--composite', default='ALL_VNIR_SWIR', help = 'output location.')

    args = parser.parse_args()

    RFACTOR = int(args.reduction_factor)
    EXTERNALDIR = Path(args.parent_directory)
    SUBDOM = int(args.subdomain_size)
    DATADIR = Path(args.data_directory)

    LABELDIR = EXTERNALDIR / args.label

    if SUBDOM % RFACTOR != 0:
        raise Exception("Subdomain size must be divisible by reduction factor.")
    if not LABELDIR.exists(): LABELDIR.mkdir(parents=True)

    BANDS = ['B05', 'B06', 'B07', 'B8A', 'B11', 'B12']

    OUTPUTDIR = LABELDIR / args.composite
    OUTPUTDIR3D = LABELDIR / f'{args.composite}_RGB'

    counter = 0
    for item in DATADIR.glob('*'):
        if not item.is_dir(): continue
        if item == 'Orlando': continue
        for SafeDir in item.glob('*.SAFE'):
            granuleDir = SafeDir / 'GRANULE'
            for SectorDir in granuleDir.glob('*'):
                ImageDir = SectorDir / 'IMG_DATA'
                MidResDir = ImageDir / f'R{RESOLUTIONS[1]}'
                fileNames = {}
                print("Currently inside: ",SectorDir)
                for file in MidResDir.glob('*jp2'):
                    band = file.name.split('_')[2]
                    passtime = datetime.strptime(file.name.split('_')[1], '%Y%m%dT%H%M%S')
                    if band in BANDS:
                        BANDDIR = LABELDIR / band
                        if not BANDDIR.exists(): BANDDIR.mkdir(parents=True)
                        with rasterio.open(file) as S2File:
                            w = S2File.width ; h = S2File.height
                            data = S2File.read(1)
                            datadown = block_reduce(data, block_size = (RFACTOR,RFACTOR), func=np.mean, cval=np.mean(data))
                            LOSUBDOM = SUBDOM // RFACTOR
                            xNumHiRes = w // SUBDOM
                            yNumHiRes = h // SUBDOM
                            xNumLowRes, yNumLowRes = [ i // LOSUBDOM for i in np.shape(datadown) ]

                            xchunks = min(xNumHiRes,xNumLowRes)
                            ychunks = min(xNumHiRes,xNumLowRes)

                            for x in range(xchunks):
                                for y in range(ychunks):
                                    # High-resolution chunks.
                                    lb = x * SUBDOM ; rb = (x+1) * SUBDOM
                                    bb = y * SUBDOM ; tb = (y+1) * SUBDOM
                                    tmpChunk = denormalize(normalize_band(data[lb:rb, bb:tb], zero=True))
                                    Chunk = np.repeat(tmpChunk[:,:,None], 3, axis=2)
                                    if np.count_nonzero(Chunk < np.uint8(0)) > 0: continue
                                    fname = f'{counter:09d}.png'

                                    im = Image.fromarray(tmpChunk, 'L')
                                    HIRESDIR = OUTPUTDIR / 'hires'
                                    if not HIRESDIR.exists(): HIRESDIR.mkdir(parents=True)
                                    filePath = HIRESDIR / fname
                                    im.save(filePath, "PNG")
                                    im.close()
                                    gc.collect()

                                    im = Image.fromarray(Chunk, "RGB")
                                    HIRESDIR = OUTPUTDIR3D / 'hires'
                                    if not HIRESDIR.exists(): HIRESDIR.mkdir(parents=True)
                                    filePath = HIRESDIR / fname
                                    print("Currently writing out to: ",filePath)
                                    im.save(filePath, "PNG")
                                    im.close()
                                    gc.collect()

                                    # Low-resolution subdomains

                                    lb = x * LOSUBDOM ; rb = (x+1) * LOSUBDOM
                                    bb = y * LOSUBDOM ; tb = (y+1) * LOSUBDOM
                                    tmpChunk = denormalize(normalize_band(datadown[lb:rb, bb:tb]))
                                    Chunk = np.repeat(tmpChunk[:,:,None], 3, axis=2)

                                    im = Image.fromarray(tmpChunk, 'L')
                                    LORESDIR = OUTPUTDIR / 'lowres'
                                    if not LORESDIR.exists(): LORESDIR.mkdir(parents=True)
                                    filePath = LORESDIR / fname
                                    im.save(filePath, "PNG")
                                    im.close()
                                    gc.collect()

                                    im = Image.fromarray(Chunk, "RGB")
                                    LORESDIR = OUTPUTDIR3D / 'lowres'
                                    if not LORESDIR.exists(): LORESDIR.mkdir(parents=True)
                                    filePath = LORESDIR / fname
                                    im.save(filePath, "PNG")
                                    im.close()
                                    gc.collect()
                                    counter += 1

if __name__ == "__main__":
    main()
