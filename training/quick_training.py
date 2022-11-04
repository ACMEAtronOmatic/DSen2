#!/usr/bin/env python3
import os ; import sys
from pathlib import Path

import tensorflow as tf
import numpy as np
import tensorflow.keras as K
import matplotlib as plt
import h5py

from argparse import ArgumentParser

DESCRIPTION = 'Training a pan-sharpening neural network.'

def normalize(image):
    return (image / 127.5) - 1.0

def decode_file(image_file):
    image = tf.io.read_file(image_file)
    image = tf.image.decode_png(img, channels=1)
    if len(image.shape) == 2:
        image = tf.expand_dims(image, axis = -1)
    return normalize(tf.cast(image, tf.float32))

def main():
    parser = ArgumentParser(description=description)
    parser.add_argument('yaml_file', help = 'YAML file with training guidelines.')
    parser.add_argument('--noclobber', action='store_true', help = 'Do not clobber existing directories.')

    args = parser.parse_args()

    noclobber = args.noclobber
    with open(args.yaml_file, 'r') as file:
        config = yaml.safe_load(file)

    RUN_ID = config['default']['run_id']
    RESTART = config['default']['restart']

    trainingHiRes = Path(config['training']['directory']['training_hires'])
    trainingLoRes = Path(config['training']['directory']['training_lowres'])

    validationHiRes = Path(config['training']['directory']['validation_hires'])
    validationLoRes = Path(config['training']['directory']['validation_lowres'])

    trainingTarget = Path(config['training']['directory']['truth_train'])
    validationTarget = Path(config['training']['directory']['truth_valid'])

    ds_train_hi = tf.data.Dataset.list_files(str(trainingHiRes / '*.png'))
    ds_train_lo = tf.data.Dataset.list_files(str(trainingLoRes / '*.png'))

    ds_valid_hi = tf.data.Dataset.list_files(str(validationHiRes / '*.png'))
    ds_valid_lo = tf.data.Dataset.list_files(str(validationLoRes / '*.png'))

    ds_target_tr = tf.data.Dataset.list_files(str(trainingTarget / '*.png'))
    ds_target_va = tf.data.Dataset.list_files(str(validationTarget / '*.png'))

    for ds in [ ds_train_hi, ds_train_lo, ds_valid_hi,
               ds_valid_lo, ds_target_tr, ds_target_va ]:
        ds.map(decode_file, num_parallel_calls = tf.data.experimental.AUTOTUNE)
        ds.batch(config['training']['dataset']['batch_size'])

    # Datasets prepped.
    # Learning rate.

    initial_lr = config['training']['initial_lr']
    epochs = config['training']['epochs']
    final_lr = config['training']['final_lr']
    steps_per_epoch = config['training']['dataset']['steps']
    if final_lr != None:
        lr_decay_factor = (final_lr / initial_lr) ** (1.0 / epochs)
        if steps_per_epoch == None: steps_per_epoch = len(ds_train_hi) # Already batched
        lr_schedule = K.optimizers.schedules.ExponentialDecay(
                initial_learning_rae = initial_lr,
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
            patience=10,
            verbose=1,
            mode='min')

    tbcb = K.callbacks.TensorBoard(
            log_dir = log_directory,
            update_freq = 'epoch')

    callbacks = [checkpointcb, tbcb]

    model = Sentinel2Model(
            scaling_factor = config['training']['scaling_factor'],
            filter_size = config['training']['filter_size'],
            num_blocks = config['training']['num_blocks'],
            interpolation = config['training']['interpolation']
            )

    optimizer = Nadam(learning_rate = lr_schedule,
                      beta_1 = config['training']['beta_1'],
                      name = 'Nadam_Sentinel2')

    model.compile(optimizer = optimizer, loss = [losses], metrics = [metrics])

    # Fit the model!

    history = model.fit(
            x = [ ds_train_hi, ds_train_lo ],
            y = ds_target_tr,
            epochs = epochs,
            verbose = 1,
            callbacks = callbacks,
            validation_data = ( [ds_valid_hi, ds_valid_lo], ds_target_va ),
            validation_steps = config['training']['validation_steps'] )

