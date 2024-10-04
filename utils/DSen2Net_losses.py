#!/usr/bin/env python3

import tensorflow as tf
import tensorflow.keras as K

@tf.keras.utils.register_keras_serializable()
class MSE_and_DSSIM(K.losses.Loss):
    def __init__(self, weight_mse = 1.0, weight_dssim = 0.3, **kwargs):
        super(MSE_and_DSSIM, self).__init__(**kwargs)
        self.weight_mse = weight_mse
        self.weight_dssim = weight_dssim

    def call(self, y_true, y_pred):
        mse = self.weight_mse * tf.reduce_mean(tf.math.square(y_pred - y_true))
        dssim = self.weight_dssim * (1. - tf.image.ssim(y_true, y_pred, max_val=1.0))
        return mse + dssim

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'weight_mse' : self.weight_mse,
            'weight_dssim' : self.weight_dssim
            })
        return config

@tf.keras.utils.register_keras_serializable()
class MAE_and_DSSIM(K.losses.Loss):
    def __init__(self, weight_mae = 1.0, weight_dssim = 0.5, max_val=1.0, **kwargs):
        super(MAE_and_DSSIM, self).__init__(**kwargs)
        self.weight_mae = weight_mae
        self.weight_dssim = weight_dssim
        self.max_val = max_val

    def call(self, y_true, y_pred):
        mae = self.weight_mae * tf.reduce_mean(tf.math.abs(y_pred - y_true),axis=[1,2,3])
        dssim = self.weight_dssim * (1. - tf.image.ssim(y_true, y_pred, max_val=self.max_val))
        return mae + dssim

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'weight_mae' : self.weight_mae,
            'weight_dssim' : self.weight_dssim,
            'max_val' : self.max_val
            })
        return config

@tf.keras.utils.register_keras_serializable()
class MAE_and_MGE(K.losses.Loss):
    def __init__(self, weight_mae = 1.0, weight_mge = 0.1, max_val = 1.0, **kwargs):
        super(MAE_and_MGE, self).__init__(**kwargs)
        self.weight_mae = weight_mae
        self.weight_mge = weight_mge
        self.max_val = max_val

    def call(self, y_true, y_pred):
        mae = self.weight_mae * tf.reduce_mean(tf.math.abs(y_pred - y_true),axis=[1,2,3])
        grads_true_x, grads_true_y = tf.unstack(tf.image.sobel_edges(y_true), num = 2, axis = -1)
        grads_pred_x, grads_pred_y = tf.unstack(tf.image.sobel_edges(y_pred), num = 2, axis = -1)
        mge = self.weight_mge * tf.reduce_mean(tf.math.squared_difference( tf.math.sqrt(tf.math.square(grads_true_x) + tf.math.square(grads_true_y)),
                                         tf.math.sqrt(tf.math.square(grads_pred_x) + tf.math.square(grads_pred_y)) ), axis = [1,2,3])
        return mae + mge

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'weight_mae' : self.weight_mae,
            'weight_mge' : self.weight_mge,
            'max_val' : self.max_val
            })
        return config
