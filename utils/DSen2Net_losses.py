#!/usr/bin/env python3

import tensorflow as tf
import tensorflow.keras as K

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
