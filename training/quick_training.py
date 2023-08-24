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
import pickle
import gc
import shutil
from datetime import datetime, timedelta

from utils.DSen2Net_new import Sentinel2Model, Sentinel2ModelUnet, Sentinel2EDSR, Sentinel2ModelSPD
from utils.DSen2Net_losses import MSE_and_DSSIM, MAE_and_DSSIM

DESCRIPTION = 'Training a pan-sharpening neural network.'

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

#def decode_file(image_file):
#    image = tf.io.read_file(image_file)
#    image = tf.image.decode_png(image, channels=1)
#    if len(image.shape) == 2:
#        image = tf.expand_dims(image, axis = -1)
#    return normalize(tf.cast(image, tf.float32))

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

def create_full_dataset(X_hi, X_low, Y_hi, X_hi_name='input_hi', X_low_name='input_low', activation='sigmoid', training=False):
    activation = activation.lower()
    ACTIVATIONS = ['sigmoid', 'tanh']
    if activation not in ACTIVATIONS: raise Exception("Activation function unknown. Options: sigmoid, tanh.")

    if training:
        rnFlip = tf.random.uniform(shape=(), maxval=1.0, dtype = tf.float64)
        rnRot  = tf.random.uniform(shape=(), maxval=4, dtype=tf.int32)
        randoms = [rnFlip, rnRot]
    else:
        randoms = [None, None]

    X_hi = X_hi.map(lambda x: decode_file(x, randoms=randoms, activation = activation), num_parallel_calls = tf.data.experimental.AUTOTUNE)
    X_low = X_low.map(lambda x: decode_file(x, randoms=randoms, activation = activation), num_parallel_calls = tf.data.experimental.AUTOTUNE)
    Y_hi = Y_hi.map(lambda x: decode_file(x, randoms=randoms, activation = activation), num_parallel_calls = tf.data.experimental.AUTOTUNE)

    ds_X = tf.data.Dataset.zip( (X_hi, X_low) ).map(lambda X_hi, X_low: {X_hi_name : X_hi, X_low_name : X_low } )
    return tf.data.Dataset.zip( (ds_X, Y_hi) )

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
    MODEL = config['default']['model'].lower()
    OPTIMIZER = config['training']['optimizer']['name'].lower()

    weightsPath = Path(config['training']['restart']['path'])

    p = Path.cwd()
    runPath = p / RUN_ID
    if runPath.exists() and noclobber:
        raise Exception("Run directory exists and noclobber set. Exiting.")
    if not runPath.exists(): runPath.mkdir(parents=True)

    yaml_name = args.yaml_file.split('/')[-1]
    shutil.copy(args.yaml_file, runPath / yaml_name)

    trainingHiRes = Path(config['training']['directory']['training_hires'])
    trainingLoRes = Path(config['training']['directory']['training_lowres'])

    validationHiRes = Path(config['training']['directory']['validation_hires'])
    validationLoRes = Path(config['training']['directory']['validation_lowres'])

    trainingTarget = Path(config['training']['directory']['truth_train'])
    validationTarget = Path(config['training']['directory']['truth_valid'])

    ds_train_hi = tf.data.Dataset.list_files(str(trainingHiRes / config['training']['glob']['training_hires']), \
                                             shuffle=False) # Definitely do NOT shuffle these datasets at read-in time
    ds_train_lo = tf.data.Dataset.list_files(str(trainingLoRes / config['training']['glob']['training_lowres']), \
                                             shuffle=False)

    ds_valid_hi = tf.data.Dataset.list_files(str(validationHiRes / config['training']['glob']['validation_hires']), shuffle=False)
    ds_valid_lo = tf.data.Dataset.list_files(str(validationLoRes / config['training']['glob']['validation_lowres']), shuffle=False)

    ds_target_tr = tf.data.Dataset.list_files(str(trainingTarget / config['training']['glob']['truth_train']), shuffle=False)
    ds_target_va = tf.data.Dataset.list_files(str(validationTarget / config['training']['glob']['truth_valid']), shuffle=False)

    ds_train = create_full_dataset(ds_train_hi, ds_train_lo, ds_target_tr,
                                   activation = config['training']['model']['activation_final'],
                                   training=config['training']['dataset']['augment'])
    ds_valid = create_full_dataset(ds_valid_hi, ds_valid_lo, ds_target_va, activation = config['training']['model']['activation_final'], training = False)

    ds_train = ds_train.shuffle(int(config['training']['dataset']['shuffle_size'])).batch(int(config['training']['dataset']['batch_size']))
    ds_valid = ds_valid.shuffle(int(config['training']['dataset']['shuffle_size'])).batch(int(config['training']['dataset']['batch_size']))

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

    # Directories for checkpoints/logs
    best_checkpoint_directory = Path(RUN_ID) / 'best_checkpoints'
    checkpoint_directory = Path(RUN_ID) / 'checkpoints'
    log_directory = Path(RUN_ID) / 'logs'
    if not checkpoint_directory.exists(): checkpoint_directory.mkdir(parents=True)
    if not log_directory.exists(): log_directory.mkdir(parents=True)
    if not best_checkpoint_directory.exists(): best_checkpoint_directory.mkdir(parents=True)

    best_checkpoint_name = best_checkpoint_directory / 'best-min_val_loss-epoch_{epoch:04d}.hdf5'
    checkpoint_name = checkpoint_directory / 'ckpt-{epoch:04d}'

    checkpointcbbest = K.callbacks.ModelCheckpoint(
            filepath = best_checkpoint_name,
            verbose = 1,
            save_weights_only = True,
            monitor = 'val_loss',
            mode = 'min',
            save_best_only = True) # Will only save best checkpoint in checkpoint format
            # VERY IMPORTANT GOTCHA HERE
            # Evidently, saving this in a non-HDF5 format with the epoch in the file name
            # will break upon weights reloading.

    checkpointcb = K.callbacks.ModelCheckpoint(
            filepath = checkpoint_name,
            verbose = 1) # PROTOBUF format


    metrics = [K.metrics.MeanAbsoluteError(),
               K.metrics.MeanAbsolutePercentageError(),
               K.metrics.MeanSquaredError()]

    #losses = [K.losses.MeanAbsoluteError()]
    losses = [MAE_and_DSSIM(weight_dssim=config['training']['fit']['weight_dssim'])]

    try:
        patience = int(config['training']['fit']['patience'])
    except:
        patience = 10
    earlystopcb = K.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            verbose=1,
            mode='min')

    tbcb = K.callbacks.TensorBoard(
            log_dir = logDir,
            update_freq = 'batch',
            write_graph = True)

    csvName = log_directory / f'training_history_{datetime.now().strftime("%Y-%m-%dT%H-%M-%S")}.csv'
    csvcb = K.callbacks.CSVLogger(str(csvName))

#   callbacks = [checkpointcb, tbcb, checkpointcbbest, earlystopcb, csvcb]
    callbacks = [tbcb, checkpointcbbest, earlystopcb, csvcb]

    # Build model.
    if MODEL == 'dsen2':
        model = Sentinel2Model(
            scaling_factor = config['training']['model']['scaling_factor'],
            filter_size = config['training']['model']['filter_size'],
            num_blocks = config['training']['model']['num_blocks'],
            interpolation = config['training']['model']['interpolation'],
            classic = config['training']['model']['classic'],
            scaling = config['training']['model']['scaling'],
            training = True
            )
    elif MODEL == 'unet':
        model = Sentinel2ModelUnet(
                scaling_factor = config['training']['model']['scaling_factor'],
                filter_sizes = config['training']['model']['unet_filter_sizes'],
                interpolation = config['training']['model']['interpolation'],
                training = True,
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
    elif MODEL == 'spd':
        model = Sentinel2ModelSPD(
                scaling_factor = config['training']['model']['scaling_factor'],
                filter_size = config['training']['model']['filter_size'],
                num_blocks = config['training']['model']['num_blocks'],
                initializer = config['training']['model']['initializer'],
                scaling = config['training']['model']['scaling'],
                training = True
                )
    else:
        raise Exception(f"Model type not eligible: {MODEL}.")


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

    if RESTART:
        if 'hdf5' in weightsPath.name.split('.')[-1].lower() or 'h5' in weightsPath.name.split('.')[-1].lower():
            model.load_weights(weightsPath)
        else:
            model.load_weights(weightsPath).expect_partial()
        startEpoch = config['training']['restart']['epoch']
    else:
        startEpoch = 0

    model.compile(optimizer = optimizer, loss = losses, metrics = metrics, jit_compile=False)

    # Fit the model!
    history = model.fit(
            ds_train,
            epochs = epochs,
            verbose = 1,
            callbacks = callbacks,
            initial_epoch = startEpoch,
            steps_per_epoch = steps_per_epoch,
            validation_data =  ds_valid,
            validation_steps = config['training']['fit']['validation_steps'] )

    savePath = runPath / 'saved_model'
    savePathKeras = runPath / 'saved_model_keras'
    savePathWeightDir = runPath / 'weights'
    savePathWeights = savePathWeightDir / f'model_{MODEL}'

    model.save(savePathKeras)           # PROTOBUF format
    model.save_weights(savePathWeights) # Looks like a tensorflow checkpoint

    histPath = runPath / 'history'
    if not histPath.exists(): histPath.mkdir(parents=True)
    with open(histPath / 'trainedHistory.pkl', 'wb') as f:
        pickle.dump(history.history, f)

    df = pd.DataFrame.from_dict(history.history)
    filePath = histPath / f'history_df.{RUN_ID}.pkl'
    df.to_pickle(filePath)

if __name__ == '__main__':
    main()
