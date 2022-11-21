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

from utils.DSen2Net_new import Sentinel2Model, Sentinel2ModelUnet
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
    return tf.data.Dataset.zip( (ds_X, Y_hi) )


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

    validationHiRes = Path(config['training']['directory']['validation_hires'])
    validationLoRes = Path(config['training']['directory']['validation_lowres'])
    validationTarget = Path(config['training']['directory']['truth_valid'])

    ds_valid_hi = tf.data.Dataset.list_files(str(validationHiRes / '*.png'), shuffle=False)
    ds_valid_lo = tf.data.Dataset.list_files(str(validationLoRes / '*.png'), shuffle=False)
    ds_target_va = tf.data.Dataset.list_files(str(validationTarget / '*.png'), shuffle=False)
    
    ds_valid = create_predict_dataset(ds_valid_hi, ds_valid_lo, ds_target_va, activation=config['training']['model']['activation_final'])
    ds_valid = ds_valid.batch(BATCH_SIZE)

    fnames = sorted(list(validationTarget.glob('*.png')))

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
    else:
        raise Exception("Model needs to be one of: dsen2, unet")
    
    if config['training']['model']['activation_final'] == 'tanh':
        denorm = denormalize_tanh
    elif config['training']['model']['activation_final'] == 'sigmoid':
        denorm = denormalize

    plot_model(model, show_shapes=True, show_layer_names=True, dpi=300, to_file = 'model_structure.png')

    print('Loading model...')
    if 'hdf5' in weightsPath.name.split('.')[-1].lower() or 'h5' in weightsPath.name.split('.')[-1].lower():
        model.load_weights(weightsPath)
    else:
        model.load_weights(weightsPath).expect_partial()
    print('Success!')

    fig = plt.figure(figsize=(10,7))
    for X, Y in ds_valid:

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

        axes = fig.subplots(nrows = 2, ncols = 3).flatten()
        images = [imhi, imlo, imtruth, imlobc, imlobl, imout]
        titles = [ '(a) VIS hi-res', '(b) NIR low-res', '(c) NIR truth', '(d) NIR bicubic', '(e) NIR bilinear', '(f) NIR CNN' ]
        
        for ax, im, title in zip(axes, images, titles):
            simple_plot(ax,im,title)
#       ax = axes[0] ; ax.imshow(imhi, cmap='bone', vmin=0.0, vmax=1.0) ; ax.set_title('(a) VIS hi-res')
#       ax = axes[1] ; ax.imshow(imlo, cmap='bone', vmin=0.0, vmax=1.0) ; ax.set_title('(b) NIR low-res')
#       ax = axes[2] ; ax.imshow(imtruth, cmap='bone', vmin=0.0, vmax=1.0) ; ax.set_title('(c) NIR truth')
#       ax = axes[3] ; ax.imshow(imlobc, cmap='bone', vmin=0.0, vmax=1.0) ; ax.set_title('(d) NIR bicubic')
#       ax = axes[4] ; ax.imshow(imlobl, cmap='bone', vmin=0.0, vmax=1.0) ; ax.set_title('(e) NIR bilinear')
#       ax = axes[5] ; ax.imshow(imout, cmap='bone', vmin=0.0, vmax=1.0) ; ax.set_title('(f) NIR CNN')

        fig.savefig('test.png', dpi=200, bbox_inches='tight')

        exit()
        fig.clf()


if __name__ == '__main__':
    main()
