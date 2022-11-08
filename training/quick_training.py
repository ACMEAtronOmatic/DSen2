#!/usr/bin/env python3
import os ; import sys
from pathlib import Path
sys.path.append('../')

import tensorflow as tf
import tensorflow.keras as K
import numpy as np
import matplotlib as plt
import pandas as pd
import h5py
from argparse import ArgumentParser
import yaml
import gc

from utils.DSen2Net_new import Sentinel2Model

DESCRIPTION = 'Training a pan-sharpening neural network.'

def create_full_dataset(X_hi, X_low, Y_hi, X_hi_name='input_hi', X_low_name='input_low'):
    ds_X = tf.data.Dataset.zip( (X_hi, X_low) ).map(lambda X_hi, X_low: {X_hi_name : X_hi, X_low_name : X_low } )
    return tf.data.Dataset.zip( (ds_X, Y_hi) )

def normalize(image):
    return image / 255.0

def decode_file(image_file):
    image = tf.io.read_file(image_file)
    image = tf.image.decode_png(image, channels=1)
    if len(image.shape) == 2:
        image = tf.expand_dims(image, axis = -1)
    return normalize(tf.cast(image, tf.float32))

def decode_file_and_upsample(image_file, scale=2):
    scale = tf.constant(scale, dtype=tf.int64)
    image = tf.io.read_file(image_file)
    image = tf.image.decode_png(image, channels=1)
    if len(image.shape) == 2:
        image = tf.expand_dims(image, axis = -1)
    w, h, c = image.shape
    wu = tf.constant(w, dtype=tf.int64) * scale ; hu = tf.constant(h, dtype=tf.int64) * scale
    image = tf.image.resize(image, [wu, hu, c])
    return normalize(tf.cast(image, tf.float32))


def create_full_dataset_2(X_hi, X_low, Y_hi, X_hi_name='input_hi', X_low_name='input_low', scale=4):
    X_hi = X_hi.map(decode_file)
    X_low = X_low.map(decode_file)
    Y_hi = Y_hi.map(decode_file)

    ds_X = tf.data.Dataset.zip( (X_hi, X_low) ).map(lambda X_hi, X_low: {X_hi_name : X_hi, X_low_name : X_low } )
    return tf.data.Dataset.zip( (ds_X, Y_hi) )


def decode_files( trainfiles, hi_true ):
    hi_train, low_train = trainfiles
    im1 = tf.io.read_file(hi_train)
    im1 = tf.image.decode_png(im1, channels=1)
    if (len(im1.shape) == 2):
        im1 = tf.expand_dims(im1, axis = -1)

    im2 = tf.io.read_file(low_train)
    im2 = tf.image.decode_png(im2, channels=1)
    if (len(im2.shape) == 2):
        im2 = tf.expand_dims(im2, axis = -1)

    im3 = tf.io.read_file(hi_true)
    im3 = tf.image.decode_png(im3, channels=1)
    if (len(im3.shape) == 2):
        im3 = tf.expand_dims(im3, axis = -1)
        
    return ( normalize(tf.cast(im1, tf.float32)), normalize(tf.cast(im2, tf.float32))) , normalize(tf.cast(im3, tf.float32))

def main():
    parser = ArgumentParser(description=DESCRIPTION)
    parser.add_argument('yaml_file', help = 'YAML file with training guidelines.')
    parser.add_argument('--noclobber', action='store_true', help = 'Do not clobber existing directories.')

    args = parser.parse_args()

    noclobber = args.noclobber
    with open(args.yaml_file, 'r') as file:
        config = yaml.safe_load(file)

    RUN_ID = config['default']['run_id']
    RESTART = config['default']['restart']

    p = Path.cwd()
    runPath = p / RUN_ID

    trainingHiRes = Path(config['training']['directory']['training_hires'])
    trainingLoRes = Path(config['training']['directory']['training_lowres'])

    validationHiRes = Path(config['training']['directory']['validation_hires'])
    validationLoRes = Path(config['training']['directory']['validation_lowres'])

    trainingTarget = Path(config['training']['directory']['truth_train'])
    validationTarget = Path(config['training']['directory']['truth_valid'])

    ds_train_hi = tf.data.Dataset.list_files(str(trainingHiRes / '*.png'), shuffle=False) # Definitely do NOT shuffle these datasets at read-in time
    ds_train_lo = tf.data.Dataset.list_files(str(trainingLoRes / '*.png'), shuffle=False)

    ds_valid_hi = tf.data.Dataset.list_files(str(validationHiRes / '*.png'), shuffle=False)
    ds_valid_lo = tf.data.Dataset.list_files(str(validationLoRes / '*.png'), shuffle=False)

    ds_target_tr = tf.data.Dataset.list_files(str(trainingTarget / '*.png'), shuffle=False)
    ds_target_va = tf.data.Dataset.list_files(str(validationTarget / '*.png'), shuffle=False)

    ds_train = create_full_dataset_2(ds_train_hi, ds_train_lo, ds_target_tr)
    ds_valid = create_full_dataset_2(ds_valid_hi, ds_valid_lo, ds_target_va)

#   ds_train = ds_train.map(decode_files, num_parallel_calls = tf.data.experimental.AUTOTUNE)
    ds_train = ds_train.shuffle(int(config['training']['dataset']['shuffle_size'])).batch(int(config['training']['dataset']['batch_size']))

#   ds_valid = ds_valid.map(decode_files, num_parallel_calls = tf.data.experimental.AUTOTUNE)
    ds_valid = ds_valid.shuffle(int(config['training']['dataset']['shuffle_size'])).batch(int(config['training']['dataset']['batch_size']))

    del ds_train_hi
    del ds_train_lo
    del ds_valid_hi
    del ds_valid_lo
    del ds_target_tr
    del ds_target_va
    gc.collect()

    # Datasets prepped.
    # Learning rate.

    initial_lr = config['training']['optimizer']['initial_lr']
    epochs = config['training']['fit']['epochs']
    final_lr = config['training']['optimizer']['final_lr']
    steps_per_epoch = config['training']['fit']['steps_per_epoch']
    if final_lr != None:
        lr_decay_factor = (final_lr / initial_lr) ** (1.0 / epochs)
        if steps_per_epoch == None: steps_per_epoch = len(ds_train) # Already batched
        lr_schedule = K.optimizers.schedules.ExponentialDecay(
                initial_learning_rate = initial_lr,
                decay_steps = steps_per_epoch,
                decay_rate = lr_decay_factor)
    else:
        lr_schedule = initial_lr

    checkpoint_directory = Path(RUN_ID) / 'checkpoints'
    log_directory = Path(RUN_ID) / 'logs'
    if not checkpoint_directory.exists(): checkpoint_directory.mkdir(parents=True)
    if not log_directory.exists(): log_directory.mkdir(parents=True)

    checkpoint_name = str(checkpoint_directory) + '/weights-{epoch:04d}-min_val_loss.hdf5'

    checkpointcb = K.callbacks.ModelCheckpoint(
            filepath = os.path.dirname(checkpoint_name),
            verbose = 1,
            save_weights_only = True,
            monitor = 'val_loss',
            mode = 'min',
            save_best_only = True)

    metrics = [K.metrics.MeanAbsoluteError(),
               K.metrics.MeanAbsolutePercentageError(),
               K.metrics.MeanSquaredError()]

    losses = [K.losses.MeanAbsoluteError()]

    earlystopcb = K.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            verbose=1,
            mode='min')

    tbcb = K.callbacks.TensorBoard(
            log_dir = log_directory,
            update_freq = 'batch')

    callbacks = [checkpointcb, tbcb]

    model = Sentinel2Model(
            scaling_factor = config['training']['model']['scaling_factor'],
            filter_size = config['training']['model']['filter_size'],
            num_blocks = config['training']['model']['num_blocks'],
            interpolation = config['training']['model']['interpolation']
            )

    optimizer = K.optimizers.Adam(learning_rate = lr_schedule,
                      beta_1 = config['training']['optimizer']['beta_1'],
                      name = 'Adam_Sentinel2')

    model.compile(optimizer = optimizer, loss = [losses], metrics = [metrics], jit_compile=False)

    # Fit the model!


    history = model.fit(
            ds_train,
            epochs = epochs,
            verbose = 1,
            callbacks = callbacks,
            steps_per_epoch = steps_per_epoch,
            validation_data =  ds_valid,
            validation_steps = config['training']['fit']['validation_steps'] )
#           validation_data =  ds_valid,
#           validation_steps = config['training']['fit']['validation_steps'] )

    df = pd.DataFrame.from_dict(history)
    histPath = runPath / 'history'
    if not histPath.exists(): histPath.mkdir(parents=True)
    filePath = histPath / f'history.{RUN_ID}.pkl'
    df.to_pickle(filePath)

if __name__ == '__main__':
    main()
