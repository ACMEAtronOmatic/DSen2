import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras.layers import Input, Conv2D, Concatenate, Add, UpSampling2D, Dropout

class Scaling(K.layers.Layer):
    def __init__(self, scale, *args, **kwargs):
        self.scale = scale
        super(Scaling, self).__init__(*args, **kwargs)

    def call(self, x, **kwargs):
        return x * self.scale

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'scale': self.scale
        })
        return config

class ResidualBlock(K.layers.Layer):
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
                Conv2D(self.filters, self.kernel_size, strides=self.strides, padding='same', kernel_initializer = self.initializer, use_bias=False),
                self.activation,
                Conv2D(self.filters, self.kernel_size, strides = self.strides, padding='same', kernel_initializer = self.initializer, use_bias=False),
                K.layers.BatchNormalization(),
                Scaling(scale=self.scaling)
                ]

            self.skip_layers = []
            if strides > 1:
                self.skip_layers = [
                        Conv2D(self.filters, 1, strides = self.strides, padding='same', kernel_initializer = self.initializer, use_bias=False),
                        K.layers.BatchNormalization()
                        ]
        else:
            self.main_layers = [
                Conv2D(self.filters, self.kernel_size, strides=self.strides, padding='same', kernel_initializer = self.initializer, use_bias=False),
                self.activation,
                Conv2D(self.filters, self.kernel_size, strides = self.strides, padding='same', kernel_initializer = self.initializer, use_bias=False),
                Scaling(scale=self.scaling)
                ]

            self.skip_layers = []
            if strides > 1:
                self.skip_layers = [
                        Conv2D(self.filters, 1, strides = self.strides, padding='same', kernel_initializer = self.initializer, use_bias=False)
                        ]

    def call(self, inputs):
        Z  = inputs
        for layer in self.main_layers:
            Z = layer(Z)
        skip_Z = inputs
        for layer in self.skip_layers:
            skip_Z = layer(skip_Z)
        return Add()([Z, skip_Z]) # Feels strange not to BatchNorm and ReLU here

    def get_config(self):
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

def Sentinel2Model(scaling_factor = 4,
                   filter_size  = 128,
                   num_blocks   = 6, 
                   interpolation = 'bilinear',
                   initializer = 'glorot_uniform',
                   filter_first = None,
                   filter_res   = None,
                   filter_last  = None,
                   training = True):

    if filter_first == None: filter_first = filter_size
    if filter_res   == None: filter_res   = filter_size
    if filter_last  == None: filter_last  = filter_size

    # Hi_Res input
    in_hi = Input(shape = [None, None, 1], name = 'input_hi')

    #Low-res input
    in_low = Input(shape = [None, None, 1], name = 'input_low')
    
    # Up-rez the input
    up_low = UpSampling2D(size = (scaling_factor, scaling_factor), interpolation = interpolation, name = 'Upsampling_low-resolution_input')(in_low)

    concat = Concatenate(name='Concatenate_two-channels')([in_hi,up_low])

    # First treat the concatenation
    x = Conv2D(filter_first, (3,3), kernel_initializer = initializer, activation='relu', padding='same', name= 'conv2d_initial')(concat)

    # Residual Blocks
    for block in range(num_blocks):
        x = ResidualBlock(filters = filter_size, initializer=initializer)(x)

    x = Dropout(0.2)(x, training=training)
    x = Conv2D(1, (3,3), kernel_initializer = initializer, padding='same', activation='tanh', name='conv2d_after_res_block')(x)

    x = Add(name='addition_final')([x, up_low])

    output = Conv2D(1, 1, kernel_initializer = initializer, padding='same', activation='sigmoid', name = 'conv2d_final')(x)
    return K.models.Model(inputs = [in_hi, in_low], outputs = output)
