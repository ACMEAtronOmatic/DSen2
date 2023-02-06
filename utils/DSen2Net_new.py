import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras.layers import Input, Conv2D, Concatenate, Add, UpSampling2D, Dropout, MaxPool2D, LeakyReLU, BatchNormalization, SpatialDropout2D
from sympy import factorint

class PixelShuffle(K.layers.Layer):
    def __init__(self,
                 block_size = 2,
                 data_format=None,
                 *args,
                 **kwargs):
        self.block_size  = block_size
        self.data_format = data_format
        super(PixelShuffle, self).__init__(*args,**kwargs)

    def call(self, inputs):
        return tf.nn.depth_to_space(inputs, block_size = self.block_size, data_format = self.data_format)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'block_size' : self.block_size,
            'data_format': self.data_format
            })
        return config

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
                Conv2D(self.filters, self.kernel_size, strides=1, padding='same', kernel_initializer = self.initializer, use_bias=False),
                self.activation,
                Conv2D(self.filters, self.kernel_size, strides = self.strides, padding='same', kernel_initializer = self.initializer, use_bias=False),
                BatchNormalization(),
                ]

            self.skip_layers = []
            if strides > 1:
                self.skip_layers = [
                        Conv2D(self.filters, 1, strides = self.strides, padding='same', kernel_initializer = self.initializer, use_bias=False),
                        BatchNormalization()
                        ]
        else:
            self.main_layers = [
                Conv2D(self.filters, self.kernel_size, strides=1, padding='same', kernel_initializer = self.initializer, use_bias=False),
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
        if self.add_batchnorm:
            return self.activation(Z + skip_Z)
        else:
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

def Sentinel2EDSR(scaling_factor = 4,
                  filter_size = 64,
                  num_blocks = 6,
                  interpolation = 'nearest',
                  initializer = 'glorot_uniform',
                  scaling = 0.1,
                  upsample = 'shuffle',
                  channels = 1,
                  add_batchnorm = False,
                  filter_first = None,
                  filter_res   = None,
                  filter_last  = None,
                  training = True):

    upsample_options = ['shuffle', 'ups']

    if upsample.lower() not in upsample_options:
        raise Exception(f"Upsampling option must be one: {upsample_options}. User provided: {upsample.lower()}")

    upsample = upsample.lower()

    if filter_first == None: filter_first = filter_size
    if filter_res   == None: filter_res   = filter_size
    if filter_last  == None: filter_last  = filter_size

    in_low = K.layers.Input( shape = [None, None, channels], name = 'input_low')
    in_hi = K.layers.Input( shape = [None, None, channels], name = 'input_hi') # Dummy layer.

    x = K.layers.Conv2D(filter_first, 3, padding='same', activation = 'relu', kernel_initializer=initializer, name = 'first_conv2d')(in_low)
    x = x_orig = K.layers.Conv2D(filter_first, 3, padding='same', activation = 'relu', kernel_initializer=initializer, name = 'second_conv2d')(x)

    for _ in range(num_blocks):
        x = ResidualBlock(filters = filter_res, initializer=initializer, scaling=scaling, add_batchnorm = add_batchnorm, name = f'ResBlock_{_:04d}')(x)

    # Global feature fusion.
    x = K.layers.Conv2D(filter_first, 1, padding='same', kernel_initializer = initializer, name = 'global_feature_fusion_01')(x)
    x = K.layers.Conv2D(filter_first, 3, padding='same', kernel_initializer = initializer, name = 'global_feature_fusion_02')(x)
    
    x = K.layers.Add(name='add_after_fusion')([x, x_orig])

    # Factorize the scaling_factor
    # The idea here is that if scaling_factor is too large, we can factorize in smaller jumps to improve performance.

    factorDict = factorint(scaling_factor)
    factorList = []
    for k, v in factorDict.items():
        for _ in range(v):
            factorList.append(k)

    if upsample == 'shuffle':
        for ii, factor in enumerate(factorList):
            x = K.layers.Conv2D(filter_last * (factor ** 2), 3, padding='same', kernel_initializer=initializer, name = f'conv2d_pixel_shuffle_{ii:02d}')(x)
            x = PixelShuffle(block_size = factor,name = f'pixel_shuffle_{ii:02d}')(x)
    elif upsample == 'ups':
        for ii, factor in enumerate(factorList):
            x = UpSampling2D(size=(factor), interpolation=interpolation, name=f'upsampling2d_{ii:02d}')(x)
            x = K.layers.Conv2D(filter_last, 3, padding='same', kernel_initializer=initializer, name = f'conv2d_upsampling2d_{ii:02d}')(x)

    output = K.layers.Conv2D(channels, 3, padding="same", kernel_initializer=initializer, name = 'final_conv2d')(x)

    return K.models.Model(inputs = [in_hi, in_low], outputs = output)

def Sentinel2Model(scaling_factor = 4,
                   filter_size  = 128,
                   num_blocks   = 6, 
                   interpolation = 'bilinear',
                   initializer = 'glorot_uniform',
                   scaling = 0.1,
                   filter_first = None,
                   filter_res   = None,
                   filter_last  = None,
                   training = True,
                   classic = True):

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
        x = ResidualBlock(filters = filter_size, initializer=initializer, scaling=scaling)(x)
        if not classic:
            x = SpatialDropout2D(0.1)(x, training=training)

    if classic:
        x = Conv2D(1, (3,3), kernel_initializer = initializer, padding='same', name = 'conv2d_final')(x)
        output = Add(name='addition_final')([x,up_low])
    else:
        x = SpatialDropout2D(0.1)(x, training=training)
        x = Conv2D(1, (3,3), kernel_initializer = initializer, padding='same', activation='tanh', name='conv2d_after_res_block')(x)
        x = Add(name='addition_final')([x, up_low])
        output = Conv2D(1, 1, kernel_initializer = initializer, padding='same', activation='sigmoid', name = 'conv2d_final')(x)

    return K.models.Model(inputs = [in_hi, in_low], outputs = output)

def Sentinel2ModelUnet(scaling_factor = 4,
                       filter_sizes = [ 64, 128, 256, 512, 1024 ],
                       interpolation = 'bilinear',
                       initializer = 'glorot_uniform',
                       activation_final = 'tanh',
                       filter_first = None,
                       filter_last  = None,
                       training = True,
                       add_batchnorm = False,
                       final_layer = 'extra'):

    if filter_first == None: filter_first = filter_sizes[0]
    if filter_last  == None: filter_last  = filter_sizes[0]

    # Hi_Res input
    in_hi = Input(shape = [None, None, 1], name = 'input_hi')

    #Low-res input
    in_low = Input(shape = [None, None, 1], name = 'input_low')
    
    # Up-rez the input
    up_low = UpSampling2D(size = (scaling_factor, scaling_factor), interpolation = interpolation, name = 'Upsampling_low-resolution_input')(in_low)

    concat = Concatenate(name='Concatenate_two-channels')([in_hi,up_low])

    x = Conv2D(filter_first, (3,3), kernel_initializer = initializer, activation='relu', padding='same', name= 'conv2d_initial')(concat)

    first = x

    # Encoder route
    skips = []
    for filt, filt_next in zip(filter_sizes[:-1], filter_sizes[1:]):
        x = ResidualBlock(filters = filt, initializer = initializer, add_batchnorm = add_batchnorm)(x)
        x = ResidualBlock(filters = filt, initializer = initializer, add_batchnorm = add_batchnorm)(x)
        skips.append(x)
        x = ResidualBlock(filters = filt_next, strides = 2, initializer = initializer, add_batchnorm = add_batchnorm)(x)

    skips = skips[::-1]

    x = Conv2D(filter_sizes[-1], (3,3), kernel_initializer=initializer, activation='relu', padding='same', name = 'bottleneck')(x)

    for filt, skip in zip(filter_sizes[::-1][1:], skips):
        x = UpSampling2D(size = (2,2), interpolation = 'nearest')(x)
        x = Conv2D(filt, (3,3), kernel_initializer = initializer, padding='same', use_bias = False)(x)
        x = LeakyReLU(0.2)(x)
        x = BatchNormalization()(x)
        x = Concatenate(name=f'concat_upsample_{filt}')([x, skip])
        x = Conv2D(filt, (3,3), kernel_initializer = initializer, padding='same', use_bias = False)(x)
        x = LeakyReLU(0.2)(x)
        x = BatchNormalization()(x)
        x = SpatialDropout2D(0.1)(x, training=training)

    x = Concatenate()([x, first])

    if final_layer == 'extra':
        x = Conv2D(filter_last, (3,3), kernel_initializer = initializer, padding='same', use_bias = False, name = 'conv2d_final')(x)
        x = LeakyReLU(0.2, name='lrelu_final')(x)
#   x = BatchNormalization(name='batchnorm_final')(x)
        output = Conv2D(1, 1, kernel_initializer = initializer, activation = activation_final, padding='same', use_bias = False, name='conv2d_output')(x)
    elif final_layer == 'simple':
        output = Conv2D(1, (3,3), kernel_initializer = initializer, activation = activation_final, padding='same', use_bias = False, name='conv2d_output')(x)
    else:
        raise Exception("Unknown final_layer. Options: extra, simple")

    return K.models.Model(inputs = [in_hi, in_low], outputs = output)
