#!/usr/bin/env python3

import tensorflow as tf
from tensorflow import keras as K

@tf.keras.utils.register_keras_serializable()
class PSNR(K.metrics.Metric):

    def __init__(self, min_val = 0.0, max_val = 1.0, name='psnr', **kwargs):
        super().__init__(name=name, **kwargs)
        self.psnr = self.add_weight(name='psnr_val', initializer = 'zeros')
        self.max_val = max_val
        self.min_val = min_val

    def update_state(self, y_true, y_pred, sample_weight = None):
        self.psnr = tf.reduce_mean(tf.image.psnr(
                tf.clip_by_value(y_true,self.min_val,self.max_val),
                tf.clip_by_value(y_pred,self.min_val,self.max_val),
                max_val = self.max_val),axis=-1)

    def result(self):
        return self.psnr

    def reset_states(self):
        self.psnr.assign(0.0)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'min_val' : self.min_val,
            'max_val' : self.max_val,
            'name' : self.name
            })
        return config

@tf.keras.utils.register_keras_serializable()
class SSIM(K.metrics.Metric):
    def __init__(self, min_val = 0.0, max_val = 1.0, name='ssim', **kwargs):
        super().__init__(name=name, **kwargs)
        self.ssim = self.add_weight(name=name, initializer = 'zeros')
        self.max_val = max_val
        self.min_val = min_val

    def update_state(self, y_true, y_pred, sample_weight = None):
        self.ssim = tf.reduce_mean(tf.image.ssim(
                tf.clip_by_value(y_true,self.min_val,self.max_val),
                tf.clip_by_value(y_pred,self.min_val,self.max_val),
                max_val = self.max_val),axis=-1)

    def result(self):
        return self.ssim

    def reset_states(self):
        self.ssim.assign(0.0)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'min_val' : self.min_val,
            'max_val' : self.max_val,
            'name' : self.name
            })
        return config

@tf.keras.utils.register_keras_serializable()
class SRE(K.metrics.Metric):
    def __init__(self, name = 'sre', remap = True, max_val = 255., min_val = 0., **kwargs):
        super().__init__(name = name, **kwargs)
        self.sre = self.add_weight(name = name, initializer = 'zeros')
        self.remap = remap
        self.max_val = max_val
        self.min_val = min_val

    def _remap(self, inputs):
        return inputs * (self.max_val - self.min_val) + self.min_val

    def update_state(self, y_true, y_pred, sample_weight = None):
        if self.remap:
            y_true = self._remap(y_true)
            y_pred = self._remap(y_pred)
        meanVal = tf.square(tf.reduce_mean(y_true, axis=[1,2,3]))
        MSE = tf.reduce_mean(tf.square(y_true - y_pred), axis=[1,2,3])
        self.sre = tf.reduce_mean( 10.0 * tf.experimental.numpy.log10(meanVal / (MSE + K.backend.epsilon())) )

    def result(self):
        return self.sre

    def reset_states(self):
        self.sre.assign(0.0)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'name' : self.name
            })
        return config

@tf.keras.utils.register_keras_serializable()
class TotalVariation(K.metrics.Metric):
    def __init__(self, name = 'total_variation', **kwargs):
        super().__init__(name = name, **kwargs)
        self.tvar = self.add_weight(name = name, initializer = 'zeros')

    def update_state(self, y_true, y_pred, sample_weight = None):
        self.tvar = tf.reduce_mean(tf.image.total_variation(y_pred))

    def result(self):
        return self.tvar

    def reset_states(self):
        self.tvar.assign(0.0)
