from __future__ import division
from keras.models import Model, Input
from keras.layers import Conv2D, Concatenate, Activation, Lambda, Add, Upsampling
import keras.backend as K

class Scaling(K.layers.Layer):
    def __init__(self, scale, *args, **kwargs):
        self.scale = scale
        super(Scaling, self).__init__(*args, **kwargs)

    def call(self, x, **kwargs):
        return x * self.scale

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'scale': self.scale,
        })
        return config

def ResidualBlock(K.layers.Layers):
    def __init__(self,
                 filters = 128,
                 kernel_size=(3,3),
                 strides = 1,
                 activation='relu',
                 initializer='glorot_uniform',
                 scaling=0.1,
                 add_batchnorm = False,
                 **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        self.filters = filters
        self.initializer = initializer
        self.strides = strides
        self.activation = K.activations.get(activation)
        self.scaling = scaling
        self.kernel_size = kernel_size
        self.add_batchnorm = add_batchnorm

        if self.add_batchnorm:
            self.main_layers = [
                Conv2D(self.filters, self.kernel_size, strides=self.strides, padding='same', initializer = self.initializer, use_bias=False),
                self.activation,
                Conv2D(self.filters, self.kernel_size, strides = self.strides, padding='same', initializer = self.initializer, use_bias=False),
                K.layers.BatchNormalization(),
                Scaling(scale=self.scaling)
                ]

            self.skip_layers = []
            if strides > 1:
                self.skip_layers = [
                        Conv2D(self.filters, 1, strides = self.strides, padding='same', initializer = self.initializer, use_bias=False),
                        K.layers.BatchNormalization()
                        ]
        else:
            self.main_layers = [
                Conv2D(self.filters, self.kernel_size, strides=self.strides, padding='same', initializer = self.initializer, use_bias=False),
                self.activation,
                Conv2D(self.filters, self.kernel_size, strides = self.strides, padding='same', initializer = self.initializer, use_bias=False),
                Scaling(scale=self.scaling)
                ]

            self.skip_layers = []
            if strides > 1:
                self.skip_layers = [
                        Conv2D(self.filters, 1, strides = self.strides, padding='same', initializer = self.initializer, use_bias=False)
                        ]

    def call(self, inputs):
        Z  = inputs
        for layer in self.main_layers:
            Z = layer(Z)
        skip_Z = inputs
        for layer in self.skip_layers:
            skip_Z = layer(skip_Z)
        return Add()([Z, skip_Z]) # Feels strange not to BatchNorm and ReLU here

    def get_config():
        config = super().get_config().copy()
        config.update({
            'filters' : self.filters,
            'initializer' : self.initializer,
            'strides' : self.strides,
            'activation' : self.activation,
            'scaling' : self.scaling,
            'kernel_size' : self.kernel_size,
            'add_batchnorm': self.add_batchnorm
            })
        return config

def resBlock(x, channels, kernel_size=[3, 3], scale=0.1):
    tmp = Conv2D(channels, kernel_size, kernel_initializer='he_uniform', padding='same')(x)
    tmp = Activation('relu')(tmp)
    tmp = Conv2D(channels, kernel_size, kernel_initializer='he_uniform', padding='same')(tmp)
    tmp = Lambda(lambda x: x * scale)(tmp)

    return Add()([x, tmp])


def s2model(input_shape, num_layers=32, feature_size=256):

    input10 = Input(shape=input_shape[0])
    input20 = Input(shape=input_shape[1])
    if len(input_shape) == 3:
        input60 = Input(shape=input_shape[2])
        x = Concatenate(axis=1)([input10, input20, input60])
    else:
        x = Concatenate(axis=1)([input10, input20])

    # Treat the concatenation
    x = Conv2D(feature_size, (3, 3), kernel_initializer='he_uniform', activation='relu', padding='same')(x)

    for i in range(num_layers):
        x = resBlock(x, feature_size)

    # One more convolution, and then we add the output of our first conv layer
    x = Conv2D(input_shape[-1][0], (3, 3), kernel_initializer='he_uniform', padding='same')(x)
    # x = Dropout(0.3)(x)
    if len(input_shape) == 3:
        x = Add()([x, input60])
        model = Model(inputs=[input10, input20, input60], outputs=x)
    else:
        x = Add()([x, input20])
        model = Model(inputs=[input10, input20], outputs=x)
    return model

def Sentinel2Model(scaling_factor = 4,
                   filter_size  = 128,
                   num_blocks   = 6, 
                   interpolation = 'bilinear',
                   filter_first = None,
                   filter_res   = None,
                   filter_last  = None):

    if filter_first == None: filter_first = filter_size
    if filter_res   == None: filter_res   = filter_size
    if filter_last  == None: filter_last  = filter_size

    # Hi_Res input
    in_hi = Input(shape = [None, None, 1], name = 'High-resolution_input')

    #Low-res input
    in_low = Input(shape = [None, None, 1], name = 'Low-resolution_input')
    
    # Up-rez the input
    in_low = Upsampling2D(size = scaling_factor, interpolation = interpolation, name = 'Upsampling_low-resolution_input')(in_low)

    assert in_low.shape == in_hi.shape

    concat = Concatenate(name='Concatenate_two-channels')([in_hi,in_low])

    # First treat the concatenation
    x = Conv2D(filter_first, (3,3), kernel_initializer = 'glorot_uniform', activation='relu', padding='same', name= 'conv2d_initial')(concat)

    # Residual Blocks
    for block in range(num_blocks):
        x = ResidualBlock(filters = filter_size)(x)

    x = Conv2D(1, (3,3), kernel_initializer = 'glorot_uniform', padding='same',name='conv2d_final')(x)

    output = Add(name='addition_final')([x, in_low])

    return Model(inputs = [in_hi, in_low], outputs = output)
