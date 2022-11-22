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
