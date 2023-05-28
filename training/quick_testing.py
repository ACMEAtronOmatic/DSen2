#!/usr/bin/env python

import os ; import sys
from pathlib import Path
sys.path.append('../')

import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras.utils import plot_model
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import h5py
from argparse import ArgumentParser
import yaml
import pickle
import gc
import shutil
import PIL
from PIL import Image
from skimage.metrics import mean_squared_error as mse, structural_similarity as ssim

from utils.DSen2Net_new import Sentinel2Model, Sentinel2ModelUnet, Sentinel2EDSR
import pygraphviz

DESCRIPTION = 'Plotting Pan-sharpening CNN output.'
BATCH_SIZE = 1

def simple_plot(ax, image, title='image', cmap='bone', vmin=0.0, vmax=1.0):
    ax.imshow(image, cmap = cmap, vmin = vmin, vmax = vmax)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])

def denormalize(image):
    return image * 255.

def denormalize_tanh(image):
    half = 255. / 2.
    return (image + 1.) * half

def normalize(image):
    return image / 255.

def normalize_tanh(image):
    half = 255. / 2.
    return (image / half) - 1.

def decode_file(image_file):
    image = tf.io.read_file(image_file)
    image = tf.image.decode_png(image, channels=1)
    if len(image.shape) == 2:
        image = tf.expand_dims(image, axis = -1)
    return normalize(tf.cast(image, tf.float32))

def decode_file_tanh(image_file):
    image = tf.io.read_file(image_file)
    image = tf.image.decode_png(image, channels=1)
    if len(image.shape) == 2:
        image = tf.expand_dims(image, axis = -1)
    return normalize_tanh(tf.cast(image, tf.float32))

def create_predict_dataset(X_hi, X_low, Y_hi, X_hi_name='input_hi', X_low_name='input_low', activation='sigmoid'):
    fname = Y_hi
    if activation.lower() == 'sigmoid':
        X_hi   = X_hi.map(decode_file, num_parallel_calls = tf.data.experimental.AUTOTUNE)
        X_low  = X_low.map(decode_file, num_parallel_calls = tf.data.experimental.AUTOTUNE)
        Y_hi   = Y_hi.map(decode_file, num_parallel_calls = tf.data.experimental.AUTOTUNE)
    elif activation.lower() == 'tanh':
        X_hi = X_hi.map(decode_file_tanh, num_parallel_calls = tf.data.experimental.AUTOTUNE)
        X_low = X_low.map(decode_file_tanh, num_parallel_calls = tf.data.experimental.AUTOTUNE)
        Y_hi = Y_hi.map(decode_file_tanh, num_parallel_calls = tf.data.experimental.AUTOTUNE)
    else:
        raise Exception("Activation function unknown. Options: sigmoid, tanh.")

    ds_X = tf.data.Dataset.zip( (X_hi, X_low) ).map(lambda X_hi, X_low: {X_hi_name : X_hi, X_low_name : X_low } )
    return tf.data.Dataset.zip( (ds_X, Y_hi, fname) )

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
    parser.add_argument('yaml_file', help='YAML file from the model run.')
    parser.add_argument('weights_path', help = 'Path for the weights that will be loaded.')

    args = parser.parse_args()

    weightsPath = Path(str(args.weights_path))

    with open(args.yaml_file, 'r') as file:
        config = yaml.safe_load(file)

    RUN_ID = config['default']['run_id']
    MODEL = config['default']['model'].lower()

    p = Path.cwd()
    runPath = p / RUN_ID
    if not runPath.exists(): runPath.mkdir(parents=True)

    imagePath = runPath / 'images'
    if not imagePath.exists(): imagePath.mkdir(parents=True)
    print(imagePath)

    validationHiRes = Path(config['inference']['directory']['test_hires'])
    validationLoRes = Path(config['inference']['directory']['test_lowres'])
    validationTarget = Path(config['inference']['directory']['truth_test'])

    ds_valid_hi = tf.data.Dataset.list_files(str(validationHiRes / config['inference']['glob']['test_hires']), shuffle=False)
    ds_valid_lo = tf.data.Dataset.list_files(str(validationLoRes / config['inference']['glob']['test_lowres']), shuffle=False)
    ds_target_va = tf.data.Dataset.list_files(str(validationTarget / config['inference']['glob']['truth_test'] ), shuffle=False)
    
    ds_valid = create_predict_dataset(ds_valid_hi, ds_valid_lo, ds_target_va, activation=config['training']['model']['activation_final'])
    ds_valid = ds_valid.batch(BATCH_SIZE)

    if MODEL == 'dsen2':
        model = Sentinel2Model(
            scaling_factor = config['training']['model']['scaling_factor'],
            filter_size = config['training']['model']['filter_size'],
            num_blocks = config['training']['model']['num_blocks'],
            interpolation = config['training']['model']['interpolation'],
            classic = config['training']['model']['classic'],
            training = False
            )
    elif MODEL == 'unet':
        model = Sentinel2ModelUnet(
                scaling_factor = config['training']['model']['scaling_factor'],
                filter_sizes = config['training']['model']['unet_filter_sizes'],
                interpolation = config['training']['model']['interpolation'],
                training = False,
                activation_final = config['training']['model']['activation_final'],
                add_batchnorm = config['training']['model']['add_batchnorm'],
                final_layer = config['training']['model']['final_layer']
                )
    elif MODEL == 'edsr':
        model = Sentinel2EDSR(
                scaling_factor = config['training']['model']['scaling_factor'],
                filter_size = config['training']['model']['filter_size'],
                num_blocks = config['training']['model']['num_blocks'],
                interpolation = config['training']['model']['interpolation'],
                upsample = config['training']['model']['upsample'],
                scaling = config['training']['model']['scaling'],
                training = True,
                initializer = config['training']['model']['initializer']
                )
    else:
        raise Exception("Model needs to be one of: dsen2, unet, edsr")
    
    if config['training']['model']['activation_final'] == 'tanh':
        denorm = denormalize_tanh
    elif config['training']['model']['activation_final'] == 'sigmoid':
        denorm = denormalize

    plot_model(model, show_shapes=True, show_layer_names=True, dpi=200, to_file = f'model_structure_{MODEL}.png')

    print('Loading model...')
    if 'hdf5' in weightsPath.name.split('.')[-1].lower() or 'h5' in weightsPath.name.split('.')[-1].lower():
        model.load_weights(weightsPath)
    else:
        model.load_weights(weightsPath).expect_partial()
    print('Success!')

    if MODEL == 'edsr':
        fig = plt.figure(figsize=(10,6))
        for X, Y, fname in ds_valid:
            hi = denorm(np.squeeze(X['input_hi'].numpy()))

    else:
        fig = plt.figure(figsize=(9,8))
        metrics = {}
        upResLogics = ['Bilinear', 'Bicubic', 'CNN']
        metricTypes = ['mae', 'mse', 'ssim']
        for upResLogic in upResLogics:
            metrics[upResLogic] = {}
            for metricType in metricTypes:
                metrics[upResLogic][metricType] = []

        for X, Y, fname in ds_valid:

            inPath = Path(str(fname.numpy()))
            fStem = inPath.stem
            outputPath = imagePath / f'{fStem}_sr.png'
    
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
    
            axes = fig.subplots(nrows = 3, ncols = 2, gridspec_kw = {'hspace': 0.2, 'wspace':0.05} ).flatten()
#           images = [imhi, imlo, imtruth, imlobc, imlobl, imout]
#           titles = [ '(a) VIS hi-res', '(b) NIR low-res', '(c) NIR truth', '(d) NIR bicubic', '(e) NIR bilinear', '(f) NIR CNN' ]
#           For OWR
            images = [imhi, imlobl, imlo, imlobc, imtruth, imout]
            titles = [ '(a) Broadband hi-res', '(b) NIR bilinear', '(c) NIR low-res', '(d) NIR bicubic', '(e) NIR truth', '(f) NIR CNN' ]
            
            for ax, im, title in zip(axes, images, titles):
                simple_plot(ax,im,title)
    
            fig.savefig(outputPath, dpi=200, bbox_inches='tight')
    
            fig.clf()

        pretty_print_metrics(metrics)

if __name__ == '__main__':
    main()
