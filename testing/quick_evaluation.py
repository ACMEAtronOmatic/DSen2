#!/usr/bin/env python3

import os ; import sys
from pathlib import Path
sys.path.append('../')

import tensorflow as tf
import tensorflow.keras as K
import pandas as pd
import numpy as np
import yaml
import json
import pickle
import gc
import shutil
import h5py
from datetime import datetime, timedelta

from argparse import ArgumentParser

from utils.DSen2Net_new import Sentinel2Model, Sentinel2ModelUnet, Sentinel2EDSR, Sentinel2ModelSPD
from utils.DSen2Net_losses import MSE_and_DSSIM, MAE_and_DSSIM
from utils.DSen2Net_metrics import PSNR, SSIM, SRE, TotalVariation

DESCRIPTION = 'Evaluating a SISR neural network.'

def normalize(image):
    return image / 255.

def normalize_tanh(image):
    half = 255. / 2.
    return (image / half) - 1.

def decode_file_tanh(image_file):
    image = tf.io.read_file(image_file)
    image = tf.image.decode_png(image, channels=1)
    if len(image.shape) == 2:
        image = tf.expand_dims(image, axis = -1)
    return normalize_tanh(tf.cast(image, tf.float32))

def decode_file(image_file, randoms = [None, None], activation = 'sigmoid'):
    image = tf.io.read_file(image_file)
    image = tf.image.decode_png(image, channels=1)

    augment = None not in randoms
    if augment:
        random_flip, random_rotate = randoms
        if random_flip > 0.5: image = tf.image.flip_left_right(image)
        image = tf.image.rot90(image, k=random_rotate)

    if len(image.shape) == 2:
        image = tf.expand_dims(image, axis = -1)
    if activation == 'sigmoid':
        return normalize(tf.cast(image,tf.float32))
    elif activation == 'tanh':
        return normalize_tanh(tf.cast(image,tf.float32))
    else:
        return tf.cast(image,tf.float32)

def create_predict_dataset(X_hi, X_low, Y_hi, X_hi_name='input_hi', X_low_name='input_low', activation='sigmoid'):
    fname = Y_hi 
    activation = activation.lower()
    ACTIVATIONS = ['sigmoid', 'tanh']
    if activation not in ACTIVATIONS: raise Exception("Activation function unknown. Options: sigmoid, tanh.")

    randoms = [None, None]

    X_hi = X_hi.map(lambda x: decode_file(x, randoms=randoms, activation = activation), num_parallel_calls = tf.data.experimental.AUTOTUNE)
    X_low = X_low.map(lambda x: decode_file(x, randoms=randoms, activation = activation), num_parallel_calls = tf.data.experimental.AUTOTUNE)
    Y_hi = Y_hi.map(lambda x: decode_file(x, randoms=randoms, activation = activation), num_parallel_calls = tf.data.experimental.AUTOTUNE)

    ds_X = tf.data.Dataset.zip( (X_hi, X_low) ).map(lambda X_hi, X_low: {X_hi_name : X_hi, X_low_name : X_low } )
    return tf.data.Dataset.zip( (ds_X, Y_hi) )

def main():
    parser = ArgumentParser(description=DESCRIPTION)
    parser.add_argument('training_directory', help = 'Location of training directory.')
    parser.add_argument('-i', '--run_ids', nargs='+', default=None, help = 'Run IDs.')

    args = parser.parse_args()

    trainingDir = Path(args.training_directory)
    runIDs = args.run_ids
    if runIDs == None:
        raise Exception("Need to provide runIDs.")

    resultsDict = {}
    for runID in runIDs:
        resultsDict[runID] = {}
        trainingPath = trainingDir / runID
        ymlFile = sorted(trainingPath.glob('*yml'))[0]
        print(ymlFile)
        with open(ymlFile, 'r') as file:
            config = yaml.safe_load(file)

        MODEL = config['default']['model'].lower()
        OPTIMIZER = config['training']['optimizer']['name'].lower()
#       BATCH_SIZE = config['inference']['dataset']['batch_size']
#       SHUFFLE_SIZE = config['inference']['dataset']['shuffle_size']
        BATCH_SIZE = 192
        SHUFFLE_SIZE = 256

        validationHiRes = Path(config['inference']['directory']['test_hires'])
        validationLoRes = Path(config['inference']['directory']['test_lowres'])
        validationTarget = Path(config['inference']['directory']['test_valid'])

        ds_valid_hi = tf.data.Dataset.list_files(str(validationHiRes / config['inference']['glob']['test_hires']), shuffle=False)
        ds_valid_lo = tf.data.Dataset.list_files(str(validationLoRes / config['inference']['glob']['test_lowres']), shuffle=False)
        ds_target_va = tf.data.Dataset.list_files(str(validationTarget / config['inference']['glob']['truth_test'] ), shuffle=False)

        ds_valid = create_predict_dataset(ds_valid_hi, ds_valid_lo, ds_target_va, activation=config['training']['model']['activation_final'])
        ds_valid = ds_valid.shuffle(SHUFFLE_SIZE).batch(BATCH_SIZE)

        initial_lr = config['training']['optimizer']['initial_lr']
        epochs = config['training']['fit']['epochs']
        final_lr = config['training']['optimizer']['final_lr']
        steps_per_epoch = config['training']['fit']['steps_per_epoch']
        if final_lr != None:
            lr_decay_factor = (final_lr / initial_lr) ** (1.0 / epochs)
            if steps_per_epoch == None: steps_per_epoch = len(ds_valid) # Already batched
            lr_schedule = K.optimizers.schedules.ExponentialDecay(
                    initial_learning_rate = initial_lr,
                    decay_steps = steps_per_epoch,
                    decay_rate = lr_decay_factor)
        else:
            lr_schedule = initial_lr

        
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
                training = False,
                initializer = config['training']['model']['initializer']
                )
        elif MODEL == 'spd':
            model = Sentinel2ModelSPD(
                scaling_factor = config['training']['model']['scaling_factor'],
                filter_size = config['training']['model']['filter_size'],
                num_blocks = config['training']['model']['num_blocks'],
                initializer = config['training']['model']['initializer'],
                scaling = config['training']['model']['scaling'],
                training = False 
                )
    # Define optimizer.
        if OPTIMIZER == 'adam':
            optimizer = K.optimizers.Adam(learning_rate = lr_schedule,
                      beta_1 = config['training']['optimizer']['beta_1'],
                      beta_2 = config['training']['optimizer']['beta_2'],
                      epsilon = config['training']['optimizer']['epsilon'],
                      name = 'Adam_Sentinel2')
        elif OPTIMIZER == 'nadam':
            optimizer = K.optimizers.Nadam(learning_rate = initial_lr,
                      beta_1 = config['training']['optimizer']['beta_1'],
                      beta_2 = config['training']['optimizer']['beta_2'],
                      epsilon = config['training']['optimizer']['epsilon'],
                      name = 'Nadam_Sentinel2')

        metrics = [K.metrics.MeanAbsoluteError(),
                   K.metrics.MeanAbsolutePercentageError(),
                   K.metrics.MeanSquaredError(),
                   PSNR(), SSIM(), SRE(scale = 255./3.), TotalVariation()]

        losses = [MAE_and_DSSIM(weight_dssim=config['training']['fit']['weight_dssim'])]

        model.compile(optimizer = optimizer, loss = losses, metrics = metrics, jit_compile=False)

        # Now, to get the best checkpoint weights.
        # Can be overriden
        weights_path = None
        if 'weights_path' in config['inference'].keys():
            weights_path = config['inference']['weights_path']

        if weights_path == None:
            bestChecksDir = trainingPath / 'best_checkpoints'
            fileList = sorted(bestChecksDir.glob('*.hdf5'))
            weightsPath = Path(fileList[-1])
        else:
            weightsPath = Path(weights_path)

        print('Loading model...')
        if 'hdf5' in weightsPath.name.split('.')[-1].lower() or 'h5' in weightsPath.name.split('.')[-1].lower():
            model.load_weights(weightsPath)
        else:
            model.load_weights(weightsPath).expect_partial()
        print('Success!')

        results = model.evaluate(ds_valid)
        resultsDict[runID]['loss'] = results[0]
        resultsDict[runID]['mae']  = results[1]
        resultsDict[runID]['mape'] = results[2]
        resultsDict[runID]['rmse'] = np.sqrt(results[3]+np.finfo(np.float32).eps)
        resultsDict[runID]['psnr'] = results[4]
        resultsDict[runID]['ssim'] = results[5]
        resultsDict[runID]['sre']  = results[6]
        resultsDict[runID]['tvar'] = results[7]
        
    with open('firewx-results.json', 'w') as f:
        json.dump(resultsDict, f, indent = 4)

if __name__ == "__main__":
    main()
