#!/usr/bin/env python3
import os ; import sys
from pathlib import Path

sys.path.append('../')

import tensorflow as tf
import tensorflow.keras as K

import numpy as np
import yaml
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from argparse import ArgumentParser
import PIL
from PIL import Image
from skimage.metrics import mean_squared_error as mse, structural_similarity as ssim
import pickle
import gc

from utils.DSen2Net_new import Sentinel2Model, Sentinel2ModelUnet, Sentinel2EDSR, Sentinel2ModelSPD
from utils.DSen2Net_losses import MSE_and_DSSIM, MAE_and_DSSIM
from utils.DSen2Net_metrics import PSNR, SSIM, SRE, TotalVariation

DESCRIPTION = ' Loading a SISR model from saved weights, compiling, and then whole-model saving.  '

def normalize(image):
    return image / 255.
        
def normalize_tanh(image):
    half = 255. / 2.
    return (image / half) - 1.

def denormalize(image):
    return image * 255.

def denormalize_tanh(image):
    half = 255. / 2.
    return (image + 1.) * half

def decode_file_tanh(image_file):
    image = tf.io.read_file(image_file)
    image = tf.image.decode_png(image, channels=1)
    if len(image.shape) == 2:
        image = tf.expand_dims(image, axis = -1)
    return normalize_tanh(tf.cast(image, tf.float32))

def decode_file(image_file):
    image = tf.io.read_file(image_file)
    image = tf.image.decode_png(image, channels=1)
    if len(image.shape) == 2:
        image = tf.expand_dims(image, axis = -1)
    return normalize(tf.cast(image, tf.float32))

def create_predict_dataset(X_hi, X_low, Y_hi, X_hi_name='input_hi', X_low_name='input_low', activation='sigmoid'):
    fname = Y_hi
    activation = activation.lower()
    ACTIVATIONS = ['sigmoid', 'tanh']
    if activation not in ACTIVATIONS: raise Exception("Activation function unknown. Options: sigmoid, tanh.")

    if activation.lower() == 'sigmoid':
        X_hi   = X_hi.map(decode_file, num_parallel_calls = tf.data.experimental.AUTOTUNE)
        X_low  = X_low.map(decode_file, num_parallel_calls = tf.data.experimental.AUTOTUNE)
        Y_hi   = Y_hi.map(decode_file, num_parallel_calls = tf.data.experimental.AUTOTUNE)
    elif activation.lower() == 'tanh':
        X_hi = X_hi.map(decode_file_tanh, num_parallel_calls = tf.data.experimental.AUTOTUNE)
        X_low = X_low.map(decode_file_tanh, num_parallel_calls = tf.data.experimental.AUTOTUNE)
        Y_hi = Y_hi.map(decode_file_tanh, num_parallel_calls = tf.data.experimental.AUTOTUNE)


    ds_X = tf.data.Dataset.zip( (X_hi, X_low) ).map(lambda X_hi, X_low: {X_hi_name : X_hi, X_low_name : X_low } )
    return tf.data.Dataset.zip( (ds_X, Y_hi, fname) )

def simple_plot(ax, image, title='image', cmap='bone', vmin=0.0, vmax=1.0):
    ax.imshow(image, cmap = cmap, vmin = vmin, vmax = vmax)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])

def pretty_print_metrics(metrics):
    print(" ------------ ")
    for imtype in metrics.keys():
        for metricType in metrics[imtype].keys():
            avgVal = np.mean(metrics[imtype][metricType])
            print(f"{imtype}, {metricType}: {avgVal}")
        print()
    print(" ------------ ")

def main():
    parser = ArgumentParser(description=DESCRIPTION)
    parser.add_argument('serialized_path', help = 'Path of serialized model.')
    parser.add_argument('validation_path', help = 'Path to testing file location.')
    parser.add_argument('-act', '--activation', default = 'sigmoid', help = 'Last activation function of model. Default: sigmoid')
    parser.add_argument('-bs', '--batch_size', default = 1, help = 'Batch size. Default: 1')

    args = parser.parse_args()
    
    serialPath = Path(str(args.serialized_path))
    validPath = Path(str(args.validation_path))
    activationFinal = str(args.activation)
    batchSize = int(args.batch_size)

    validationHiRes = validPath / 'red'
    validationLoRes = validPath / 'nirLow'
    validationTarget = validPath / 'nirHigh'

    ds_valid_hi = tf.data.Dataset.list_files(str(validationHiRes / '*.png'), shuffle=False)
    ds_valid_lo = tf.data.Dataset.list_files(str(validationLoRes / '*.png'), shuffle=False)
    ds_target_va = tf.data.Dataset.list_files(str(validationTarget / '*.png'), shuffle=False)

    print("Creating validation dataset.")
    ds_valid = create_predict_dataset(ds_valid_hi, ds_valid_lo, ds_target_va, activation = activationFinal)

    ds_valid = ds_valid.batch(batchSize)

    print("Loading serialized model.")
    model = K.saving.load_model(serialPath, compile = False)
    print('Success!')

    metrics = {}
    upResLogics = ['Bilinear', 'Bicubic', 'CNN']
    metricTypes = ['mae', 'mse', 'ssim']

    for upResLogic in upResLogics:
        metrics[upResLogic] = {}
        for metricType in metricTypes:
            metrics[upResLogic][metricType] = []

    if activationFinal.lower() == 'sigmoid':
        denorm = denormalize
    else:
        denorm = denormalize_tanh

    fig = plt.figure(figsize=(8,6))
    for X, Y, F in ds_valid:
        inPath = Path(str(F.numpy()))
        fStem = inPath.stem
        outputPath = f'{fStem}_sr.png'

        hi = denorm(np.squeeze(X['input_hi'].numpy()))
        lo = denorm(np.squeeze(X['input_low'].numpy()))
        out = denorm(np.squeeze(model.predict(X)))
        truth = denorm(np.squeeze(Y.numpy()))
        # Create images out of all these.
        imhi = Image.fromarray(hi)
        imlo = Image.fromarray(lo)
        imout = Image.fromarray(out)
        imtruth = Image.fromarray(truth)

        hhi = imhi.height ; whi = imhi.width
        hlo = imlo.height ; hlo = imlo.width

        imlobc = imlo.resize( [whi, hhi], resample = Image.Resampling.BICUBIC)
        imlobl = imlo.resize( [whi, hhi], resample = Image.Resampling.BILINEAR)

        # Compute metrics
        metrics['Bilinear']['mae'].append(np.mean(np.abs(np.array(imlobl) - np.array(imtruth))))
        metrics['Bilinear']['mse'].append(mse(np.array(imtruth),np.array(imlobl)))
        metrics['Bilinear']['ssim'].append(ssim(np.array(imtruth),np.array(imlobl), data_range = np.array(imlobl).max() - np.array(imlobl).min()))
        metrics['Bicubic']['mae'].append(np.mean(np.abs(np.array(imlobc) - np.array(imtruth))))
        metrics['Bicubic']['mse'].append(mse(np.array(imtruth),np.array(imlobc)))
        metrics['Bicubic']['ssim'].append(ssim(np.array(imtruth),np.array(imlobc), data_range = np.array(imlobc).max() - np.array(imlobc).min()))
        metrics['CNN']['mae'].append(np.mean(np.abs(np.array(imout) - np.array(imtruth))))
        metrics['CNN']['mse'].append(mse(np.array(imtruth),np.array(imout)))
        metrics['CNN']['ssim'].append(ssim(np.array(imtruth),np.array(imout), data_range = np.array(imout).max() - np.array(imout).min()))

        # Plot images
        axes = fig.subplots(nrows = 2, ncols = 3, gridspec_kw = {'hspace': 0.2, 'wspace':0.05} ).flatten()
        images = [imhi, imlo, imtruth, imlobc, imlobl, imout]
        titles = [ '(a) VIS hi-res', '(b) NIR low-res', '(c) NIR truth', '(d) NIR bicubic', '(e) NIR bilinear', '(f) NIR CNN' ]
#
        for ax, im, title in zip(axes, images, titles):
            simple_plot(ax,im,title)
#
        fig.savefig(outputPath, dpi=200, bbox_inches='tight')

        fig.clf()

if __name__ == '__main__':
    main()
