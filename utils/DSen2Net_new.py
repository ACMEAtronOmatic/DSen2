import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras.layers import Input, Conv2D, Concatenate, Add, UpSampling2D, Dropout, MaxPool2D, LeakyReLU, BatchNormalization, SpatialDropout2D
from sympy import factorint

class SPDLayer(K.layers.Layer):
    def __init__(self,
                 block_size = 2,
                 data_format = 'NHWC',
                 *args,
                 **kwargs):
        self.block_size  = block_size
        self.data_format = data_format
        super(SPDLayer, self).__init__(*args,**kwargs)

    def call(self, inputs):
        return tf.nn.space_to_depth(inputs, block_size = self.block_size, data_format = self.data_format)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'block_size' : self.block_size,
            'data_format' : self.data_format
            })
        return config


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

class BasicBlock(K.layers.Layer):
    def __init__(self,
                 filters = 32,
                 kernel_size = (3,3),
                 strides = 1,
                 use_bias = False,
                 spectral_normalization = False,
                 alpha = 0.3,
                 scaling = 1.0,
                 **kwargs):
        super(BasicBlock, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides,
        self.use_bias = use_bias
        self.spectral_normalization = spectral_normalization
        self.alpha = alpha

        self.conv2d = K.layers.Conv2D(filters = self.filters,
                                      kernel_size = self.kernel_size,
                                      strides = self.strides,
                                      use_bias = use_bias,
                                      activation = None)
        self.prelu = K.layers.PReLU(alpha_initializer  = K.initializers.Constant(value = self.alpha))

    def build(self, input_shape, **kwargs):
        self.gamma = self.add_weight(
                shape = (),
                initializer = K.initializers.Constant(value = self.scaling),
                trainable = True
                )
        super(BasicBlock, self).build(input_shape, **kwargs)

    def call(self, inputs):
        x = self.conv2d(inputs)
        x = self.prelu(x)
        return x * self.gamma

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'filters' : self.filters,
            'kernel_size' : self.kernel_size,
            'strides' : self.strides,
            'use_bias' : self.use_bias,
            'spectral_normalization' : self.spectral_normalization,
            'alpha' : self.alpha,
            'scaling' : self.scaling
            })
        return config

class ResidualDenseBlock(K.layers.Layer):
    def __init__(self,
                 num_blocks = 5,
                 block_filters = 64,
                 block_kernel_size = (3,3),
                 block_strides = 1,
                 block_scaling = 1.0,
                 block_alpha = 0.3,
                 block_use_bias = False,
                 block_spectral_normalization = False,
                 scaling = 0.2,
                 filters = 64,
                 kernel_size = (3,3),
                 strides = 1,
                 use_bias = False,
                 spectral_normalization = False,
                 **kwargs):
        super(ResidualDenseBlock, self).__init__(**kwargs)
        # Number of blocks
        self.num_blocks = num_blocks

        # Values of convolutions in the basic block
        self.block_filters = filters
        self.block_kernel_size = kernel_size
        self.block_strides = strides
        self.block_use_bias = use_bias
        self.block_spectral_normalization = spectral_normalization
        self.block_alpha = block_alpha
        self.block_scaling = block_scaling

        # Values for the final convolution of the RDB plus initial scaling value
        self.scaling = scaling
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.use_bias = use_bias
        self.spectral_normalization = spectral_normalization

        self.blocks = []
        for _ in range(self.num_blocks):
            self.blocks.append(BasicBlock(filters = self.block_filters,
                                     kernel_size = self.kernel_size,
                                     strides = self.block_strides,
                                     use_bias = self.block_use_bias,
                                     spectral_normalization = self.block_spectral_normalization,
                                     alpha = self.block_alpha,
                                     scaling = self.block_scaling))

        self.final_conv = K.layers.Conv2D(filters = self.filters,
                                          kernel_size = self.kernel_size,
                                          strides = self.strides,
                                          use_bias = self.use_bias)
        self.concat = K.layers.Concatenate(axis=-1)

    def build(self, input_shape):
        self.beta = self.add_weight(
                shape = (),
                initializer = K.initializers.Constant(value = self.scaling),
                trainable = True
                )
        super(ResidualDenseBlock, self).build(input_shape)

    def call(self, inputs):
        cc = [inputs]
        for block in self.blocks:
            x = block(self.concat(cc))
            cc.append(x)
        x = self.concat(cc)
        out = self.final_conv(x)
        return K.layers.Add()([out * self.beta, inputs])

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_blocks' : self.num_blocks,
            'scaling' : self.scaling,
            'filters' : self.filters,
            'kernel_size' : self.kernel_size,
            'strides' : self.strides,
            'use_bias' : self.use_bias,
            'spectral_normalization' : self.spectral_normalization,
            'block_scaling' : self.block_scaling,
            'block_alpha' : self.block_alpha,
            'block_filters' : self.block_filters,
            'block_kernel_size' : self.block_kernel_size,
            'block_strides' : self.block_strides,
            'block_use_bias' : self.block_use_bias,
            'block_spectral_normalization' : self.block_spectral_normalization
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
#           self.main_layers = [
#               Conv2D(self.filters, self.kernel_size, strides=1, padding='same', kernel_initializer = self.initializer, use_bias=False),
#               self.activation,
#               Conv2D(self.filters, self.kernel_size, strides = self.strides, padding='same', kernel_initializer = self.initializer, use_bias=False),
#               BatchNormalization(),
#               ]

            self.skip_layers = []
            if strides > 1:
                self.skip_layers = [
                        Conv2D(self.filters, 1, strides = self.strides, padding='same', kernel_initializer = self.initializer, use_bias=False),
                        BatchNormalization()
                        ]
            self.main_layers = [
                    BatchNormalization(),
                    self.activation,
                    Conv2D(self.filters, self.kernel_size, strides = self.strides, padding = 'same', kernel_initializer = self.initializer, use_bias = False),
                    BatchNormalization(),
                    self.activation,
                    Conv2D(self.filters, self.kernel_size, strides = 1, padding = 'same', kernel_initializer = self.initializer, use_bias = False)
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

    upsample_options = ['shuffle', 'upsample']

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
    elif upsample == 'upsample':
        x = K.layers.Conv2D(filter_last, 3, padding='same', kernel_initializer=initializer, name = f'conv2d_upsampling2d_first')(x)
        for ii, factor in enumerate(factorList):
            x = UpSampling2D(size=(factor), interpolation=interpolation, name=f'upsampling2d_{ii:02d}')(x)
            x = K.layers.Conv2D(filter_last, 3, padding='same', kernel_initializer=initializer, name = f'conv2d_upsampling2d_{ii:02d}')(x)

    output = K.layers.Conv2D(channels, 3, padding="same", kernel_initializer=initializer, name = 'final_conv2d')(x)

    return K.models.Model(inputs = [in_hi, in_low], outputs = output)

def Sentinel2Model(scaling_factor = 4,
                   filter_size  = 128,
                   num_blocks   = 6, 
                   interpolation = 'bilinear',
                   initializer = 'truncated_normal',
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
        x = ResidualBlock(filters = filter_size, initializer=initializer, scaling=scaling, add_batchnorm = False)(x)
#       if not classic:
#           x = SpatialDropout2D(0.1)(x, training=training)

    if classic:
        x = Conv2D(1, (3,3), kernel_initializer = initializer, padding='same', name = 'conv2d_final')(x)
        output = Add(name='addition_final')([x,up_low])
    else:
        x = Conv2D(1, (3,3), kernel_initializer = initializer, padding='same', activation='tanh', name='conv2d_after_res_block')(x)
        x = Add(name='addition_final')([x, up_low])
        output = Conv2D(1, 1, kernel_initializer = initializer, padding='same', activation='sigmoid', name = 'conv2d_final')(x)

    return K.models.Model(inputs = [in_hi, in_low], outputs = output)

def Sentinel2ModelSPD(scaling_factor = 4,
                      filter_size  = 128,
                      num_blocks   = 6, 
                      initializer = 'truncated_normal',
                      scaling = 0.1,
                      channels = 1,
                      filter_first = None,
                      filter_res   = None,
                      filter_last  = None,
                      training = True,
                      classic = True):

    if filter_first == None: filter_first = filter_size
    if filter_res   == None: filter_res   = filter_size
    if filter_last  == None: filter_last  = filter_size

    # Hi_Res input
    in_hi = Input(shape = [None, None, channels], name = 'input_hi')

    #Low-res input
    in_low = Input(shape = [None, None, channels], name = 'input_low')

    # Reduce the Hi-res input to the low-res input
    down_hi = SPDLayer(block_size = scaling_factor, name = 'space_to_depth_hi-res')(in_hi)
#   down_hi = Conv2D(filter_first, (scaling_factor*2-1,scaling_factor*2-1), stride=scaling_factor, activation='relu', padding='same', name = 'strided_conv2d_hi-res')(in_hi)

    # Concatenate
    concat = Concatenate(name='Concatenate_two-channels')([down_hi,in_low])

    # First treat the concatenation
    x = x_orig = Conv2D(filter_first, (3,3), kernel_initializer = initializer, activation='relu', padding='same', name= 'conv2d_initial')(concat)

    # Residual Blocks
    for block in range(num_blocks):
        x = ResidualBlock(filters = filter_size, initializer=initializer, scaling=scaling)(x)
#       if not classic:
#           x = SpatialDropout2D(0.1)(x, training=training)

    # Global feature fusion -- get number of filters to match
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

    if classic:
        for ii, factor in enumerate(factorList):
            x = K.layers.Conv2D(filter_last * (factor ** 2), 3, padding='same', kernel_initializer=initializer, name = f'conv2d_pixel_shuffle_{ii:02d}')(x)
            x = PixelShuffle(block_size = factor,name = f'pixel_shuffle_{ii:02d}')(x)
    else:
        x = K.layers.Conv2D(filter_last, 3, padding='same', kernel_initializer=initializer, name = f'conv2d_upsampling2d_first')(x)
        for ii, factor in enumerate(factorList):
            x = UpSampling2D(size=(factor), interpolation=interpolation, name=f'upsampling2d_{ii:02d}')(x)
            x = K.layers.Conv2D(filter_last, 3, padding='same', kernel_initializer=initializer, name = f'conv2d_upsampling2d_{ii:02d}')(x)
    output = K.layers.Conv2D(channels, 3, padding="same", kernel_initializer=initializer, name = 'final_conv2d')(x)

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
