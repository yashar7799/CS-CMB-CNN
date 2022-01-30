# Copyright 2021 The AI-Medic\Cell-Segmentation Authors. All Rights Reserved.
# License stuff will be written here later...

"""DeepLabV3PlusAttention(using Resnet 50 as backbone) model implemented from scratch using tensorflow.keras

DeepLabV3Plus is a new architecture and has many use cases in image segmentation tasks.
also attention is one of the novel method to increase your model performance.

It supports various image sizes starting from 32*32

Encoder part of architecture is a Resnet 50, pretarin on imagenet.

In the table below you can see multiple performances resulted from different configurations
of UNet:

+------------------------+---------------------+---------------------+--------------+
| Model Name             |   DSB-2018 Val Dice |   DSB-2018 Val Loss | Params (M)   |
+========================+=====================+=====================+==============+
| DeepLabV3PlusAttention |               82.27 |              0.2701 | 57,451,221   |
+------------------------+---------------------+---------------------+--------------+

* DSB-2018 : DATA-SCIENCE-BOWL-2018

Reference:
  - [code](https://github.com/rezazad68/AttentionDeeplabv3p)
  - [paper](https://link.springer.com/chapter/10.1007/978-3-030-66415-2_16)
  - [paper](https://openreview.net/pdf?id=oMQOHIS1JWy)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.keras.optimizers import Adam
import os
import warnings
import numpy as np
import cv2
from tensorflow.keras import backend as K

from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.layers import Lambda, Softmax, SeparableConv2D, MaxPooling2D, Reshape, Concatenate, Dense, \
    Activation, Input, GlobalMaxPooling2D, Dropout, concatenate, ConvLSTM2D, BatchNormalization, Conv2D, Dense, \
    multiply, concatenate, Conv3D, GlobalAveragePooling2D, ZeroPadding2D, DepthwiseConv2D, AveragePooling2D, Layer, \
    InputSpec
from tensorflow.keras.utils import get_source_inputs
from tensorflow.keras import backend as K
from tensorflow.keras.applications import imagenet_utils
from tensorflow.python.keras.utils import conv_utils
from tensorflow import keras
from tensorflow.keras.utils import get_file
from tensorflow.keras.layers import Add
import tensorflow as tf

TF_WEIGHTS_PATH = "https://github.com/bonlime/keras-deeplab-v3-plus/releases/download/1.1/deeplabv3_xception_tf_dim_ordering_tf_kernels.h5"


class BilinearUpsampling(Layer):
    """Just a simple bilinear upsampling layer. Works only with TF.
       Args:
           upsampling: tuple of 2 numbers > 0. The upsampling ratio for h and w
           output_size: used instead of upsampling arg if passed!
    """

    def __init__(self, upsampling=(2, 2), output_size=None, l_name=None, data_format=None, **kwargs):

        super(BilinearUpsampling, self).__init__(**kwargs)

        self.data_format = conv_utils.normalize_data_format(data_format)
        # self.name = l_name
        self.input_spec = InputSpec(ndim=4)
        if output_size:
            self.upsample_size = conv_utils.normalize_tuple(
                output_size, 2, 'size')
            self.upsampling = None
        else:
            self.upsampling = conv_utils.normalize_tuple(upsampling, 2, 'size')

    def compute_output_shape(self, input_shape):
        if self.upsampling:
            height = self.upsampling[0] * \
                     input_shape[1] if input_shape[1] is not None else None
            width = self.upsampling[1] * \
                    input_shape[2] if input_shape[2] is not None else None
        else:
            height = self.upsample_size[0]
            width = self.upsample_size[1]
        return (input_shape[0],
                height,
                width,
                input_shape[3])

    def call(self, inputs):
        if self.upsampling:
            return tf.compat.v1.image.resize_bilinear(inputs, (inputs.shape[1] * self.upsampling[0],
                                                               inputs.shape[2] * self.upsampling[1]),
                                                      align_corners=True)
            # return tf.image.resize_bilinear(inputs, (inputs.shape[1] * self.upsampling[0],
            #                                            inputs.shape[2] * self.upsampling[1]),
            #                                   align_corners=True )
        else:
            return tf.compat.v1.image.resize_bilinear(inputs, (self.upsample_size[0],
                                                               self.upsample_size[1]),
                                                      align_corners=True)
            # return tf.image.resize_bilinear(inputs, (self.upsample_size[0],
            #                                            self.upsample_size[1]),
            #                                   align_corners=True )

    def get_config(self):
        config = {'size': self.upsampling,
                  'data_format': self.data_format}
        base_config = super(BilinearUpsampling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def SepConv_BN(x, filters, prefix, stride=1, kernel_size=3, rate=1, depth_activation=False, epsilon=1e-3):
    """ SepConv with BN between depthwise & pointwise. Optionally add activation after BN
        Implements right "same" padding for even kernel sizes
        Args:
            x: input tensor
            filters: num of filters in pointwise convolution
            prefix: prefix before name
            stride: stride at depthwise conv
            kernel_size: kernel size for depthwise convolution
            rate: atrous rate for depthwise convolution
            depth_activation: flag to use activation between depthwise & poinwise convs
            epsilon: epsilon to use in BN layer
    """

    if stride == 1:
        depth_padding = 'same'
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        x = ZeroPadding2D((pad_beg, pad_end))(x)
        depth_padding = 'valid'

    if not depth_activation:
        x = Activation('relu')(x)
    x = DepthwiseConv2D((kernel_size, kernel_size), strides=(stride, stride), dilation_rate=(rate, rate),
                        padding=depth_padding, use_bias=False, name=prefix + '_depthwise')(x)
    x = BatchNormalization(name=prefix + '_depthwise_BN', epsilon=epsilon)(x)
    if depth_activation:
        x = Activation('relu')(x)
    x = Conv2D(filters, (1, 1), padding='same',
               use_bias=False, name=prefix + '_pointwise')(x)
    x = BatchNormalization(name=prefix + '_pointwise_BN', epsilon=epsilon)(x)
    if depth_activation:
        x = Activation('relu')(x)

    return x


def conv2d_same(x, filters, prefix, stride=1, kernel_size=3, rate=1):
    """Implements right 'same' padding for even kernel sizes
        Without this there is a 1 pixel drift when stride = 2
        Args:
            x: input tensor
            filters: num of filters in pointwise convolution
            prefix: prefix before name
            stride: stride at depthwise conv
            kernel_size: kernel size for depthwise convolution
            rate: atrous rate for depthwise convolution
    """
    if stride == 1:
        return Conv2D(filters,
                      (kernel_size, kernel_size),
                      strides=(stride, stride),
                      padding='same', use_bias=False,
                      dilation_rate=(rate, rate),
                      name=prefix)(x)
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        x = ZeroPadding2D((pad_beg, pad_end))(x)
        return Conv2D(filters,
                      (kernel_size, kernel_size),
                      strides=(stride, stride),
                      padding='valid', use_bias=False,
                      dilation_rate=(rate, rate),
                      name=prefix)(x)


def xception_block(inputs, depth_list, prefix, skip_connection_type, stride,
                   rate=1, depth_activation=False, return_skip=False):
    """ Basic building block of modified Xception network
        Args:
            inputs: input tensor
            depth_list: number of filters in each SepConv layer. len(depth_list) == 3
            prefix: prefix before name
            skip_connection_type: one of {'conv','sum','none'}
            stride: stride at last depthwise conv
            rate: atrous rate for depthwise convolution
            depth_activation: flag to use activation between depthwise & pointwise convs
            return_skip: flag to return additional tensor after 2 SepConvs for decoder
            """
    residual = inputs
    for i in range(3):
        residual = SepConv_BN(residual,
                              depth_list[i],
                              prefix + '_separable_conv{}'.format(i + 1),
                              stride=stride if i == 2 else 1,
                              rate=rate,
                              depth_activation=depth_activation)
        if i == 1:
            skip = residual
    if skip_connection_type == 'conv':
        shortcut = conv2d_same(inputs, depth_list[-1], prefix + '_shortcut',
                               kernel_size=1,
                               stride=stride)
        shortcut = BatchNormalization(name=prefix + '_shortcut_BN')(shortcut)
        outputs = layers.add([residual, shortcut])
    elif skip_connection_type == 'sum':
        outputs = layers.add([residual, inputs])
    elif skip_connection_type == 'none':
        outputs = residual
    if return_skip:
        return outputs, skip
    else:
        return outputs


class DeepLabV3PlusAttention:
    """
    DeepLabV3Plus class
    Instantiates the DeepLabV3Plus architecture.

    Reference:
        - [Github source](https://github.com/rezazad68/AttentionDeeplabv3p)

    For image segmentation use cases, see
      [this page for detailed examples](
        https://keras.io/examples/vision/deeplabv3_plus/)
    """

    def __init__(self, input_shape=(256, 256, 3), **kwargs):
        """
        Parameters
        ----------
        input_shape: shape tuple, in "channels_last" format;
            it should have exactly 3 inputs channels, and width and
            height should be no smaller than 32.
            E.g. `(256, 256, 3)` would be one valid value. Default to `None`.
        backbone: string
            it should have a value from this set: ['xception', 'resnet50',  'resnet101', 'mobilenetv2']
        OS: int
            output stride
            commonly take value 8, 16
        num_classes: int
            in segmentation determine number of label in each pixel (pixel classification)
        """
        self.input_shape = input_shape
        self.OS = 16
        self.input_tensor = None
        self.weights = 'pascal_voc'
        self.classes = 1

    def get_model(self) -> Model:
        """
        This method returns a Keras image segmentation model.

        Returns
        -------
        A `Tensorflow.keras.Model` instance.
        """
        weights = self.weights
        input_tensor = self.input_tensor
        input_shape = self.input_shape
        classes = 1
        if not (weights in {'pascal_voc', None}):
            raise ValueError('The `weights` argument should be either '
                             '`None` (random initialization) or `pascal_voc` '
                             '(pre-trained on PASCAL VOC)')

        if K.backend() != 'tensorflow':
            raise RuntimeError('The Deeplabv3+ model is only available with '
                               'the TensorFlow backend.')

        if self.OS == 8:
            entry_block3_stride = 1
            middle_block_rate = 2  # ! Not mentioned in paper, but required
            exit_block_rates = (2, 4)
            atrous_rates = (12, 24, 36)
        else:
            entry_block3_stride = 2
            middle_block_rate = 1
            exit_block_rates = (1, 2)
            atrous_rates = (6, 12, 18)

        if input_tensor is None:
            img_input = Input(shape=input_shape)
        else:
            if not K.is_keras_tensor(input_tensor):
                img_input = Input(tensor=input_tensor, shape=input_shape)
            else:
                img_input = input_tensor

        x = Conv2D(32, (3, 3), strides=(2, 2),
                   name='entry_flow_conv1_1', use_bias=False, padding='same')(img_input)
        x = BatchNormalization(name='entry_flow_conv1_1_BN')(x)
        x = Activation('relu')(x)

        x = conv2d_same(x, 64, 'entry_flow_conv1_2', kernel_size=3, stride=1)
        x = BatchNormalization(name='entry_flow_conv1_2_BN')(x)
        x = Activation('relu')(x)

        x = xception_block(x, [128, 128, 128], 'entry_flow_block1',
                           skip_connection_type='conv', stride=2,
                           depth_activation=False)
        x, skip1 = xception_block(x, [256, 256, 256], 'entry_flow_block2',
                                  skip_connection_type='conv', stride=2,
                                  depth_activation=False, return_skip=True)

        x = xception_block(x, [256, 256, 728], 'entry_flow_block3',
                           skip_connection_type='conv', stride=entry_block3_stride,
                           depth_activation=False)
        for i in range(16):
            x = xception_block(x, [256, 256, 728], 'middle_flow_unit_{}'.format(i + 1),
                               skip_connection_type='sum', stride=1, rate=middle_block_rate,
                               depth_activation=False)

        x = xception_block(x, [728, 728, 728], 'exit_flow_block1',
                           skip_connection_type='conv', stride=1, rate=exit_block_rates[0],
                           depth_activation=False)
        x = xception_block(x, [1024, 1024, 1024], 'exit_flow_block2',
                           skip_connection_type='none', stride=1, rate=exit_block_rates[1],
                           depth_activation=True)
        # end of feature extractor

        # branching for Atrous Spatial Pyramid Pooling
        # simple 1x1
        b0 = Conv2D(256, (1, 1), padding='same', use_bias=False, name='aspp0')(x)
        b0 = BatchNormalization(name='aspp0_BN', epsilon=1e-5)(b0)
        b0 = Activation('relu', name='aspp0_activation')(b0)

        # rate = 6 (12)
        b1 = SepConv_BN(x, 256, 'aspp1',
                        rate=atrous_rates[0], depth_activation=True, epsilon=1e-5)
        # rate = 12 (24)
        b2 = SepConv_BN(x, 256, 'aspp2',
                        rate=atrous_rates[1], depth_activation=True, epsilon=1e-5)
        # rate = 18 (36)
        b3 = SepConv_BN(x, 256, 'aspp3',
                        rate=atrous_rates[2], depth_activation=True, epsilon=1e-5)

        # Image Feature branch
        out_shape = int(np.ceil(input_shape[0] / self.OS))
        b4 = AveragePooling2D(pool_size=(out_shape, out_shape))(x)
        b4 = Conv2D(256, (1, 1), padding='same',
                    use_bias=False, name='image_pooling')(b4)
        b4 = BatchNormalization(name='image_pooling_BN', epsilon=1e-5)(b4)
        b4 = Activation('relu')(b4)
        b4 = BilinearUpsampling((out_shape, out_shape), l_name='up1')(b4)
        b0_1 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(b0)
        b0_1 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(b0_1)
        b0_1 = Dropout(0.5)(b0_1)
        b0_c = concatenate([b0, b0_1], axis=3)
        b0_2 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(b0_c)
        b0 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(b0_2)

        # b1
        b1_1 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(b1)
        b1_1 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(b1_1)
        b1_1 = Dropout(0.5)(b1_1)
        b1_c = concatenate([b1, b1_1], axis=3)
        b1_2 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(b1_c)
        b1 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(b1_2)

        # b2
        b2_1 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(b2)
        b2_1 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(b2_1)
        b2_1 = Dropout(0.5)(b2_1)
        b2_c = concatenate([b2, b2_1], axis=3)
        b2_2 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(b2_c)
        b2 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(b2_2)

        # b3
        b3_1 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(b3)
        b3_1 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(b3_1)
        b3_1 = Dropout(0.5)(b3_1)
        b3_c = concatenate([b3, b3_1], axis=3)
        b3_2 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(b3_c)
        b3 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(b3_2)

        # b4
        b4_1 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(b4)
        b4_1 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(b4_1)
        b4_1 = Dropout(0.5)(b4_1)
        b4_c = concatenate([b4, b4_1], axis=3)
        b4_2 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(b4_c)
        b4 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(b4_2)
        Dense0 = Dense(256, activation='relu', kernel_initializer='he_normal', use_bias=False)
        Dense1 = Dense(32, activation='relu', kernel_initializer='he_normal', use_bias=False)
        Dense2 = Dense(256, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)

        b0_1 = Reshape((1, 1, 256))(GlobalAveragePooling2D()(b0))
        b0_1 = Dense2(Dense1(Dense0(b0_1)))

        b1_1 = Reshape((1, 1, 256))(GlobalAveragePooling2D()(b1))
        b1_1 = Dense2(Dense1(Dense0(b1_1)))

        b2_1 = Reshape((1, 1, 256))(GlobalAveragePooling2D()(b2))
        b2_1 = Dense2(Dense1(Dense0(b2_1)))

        b3_1 = Reshape((1, 1, 256))(GlobalAveragePooling2D()(b3))
        b3_1 = Dense2(Dense1(Dense0(b3_1)))

        b4_1 = Reshape((1, 1, 256))(GlobalAveragePooling2D()(b4))
        b4_1 = Dense2(Dense1(Dense0(b4_1)))

        x0 = multiply([b0, b0_1])
        x1 = multiply([b1, b1_1])
        x2 = multiply([b2, b2_1])
        x4 = multiply([b4_1, b4])
        x3 = multiply([b3, b3_1])

        x0 = Reshape((16, 16, 1, 256))(x0)
        x1 = Reshape((16, 16, 1, 256))(x1)
        x2 = Reshape((16, 16, 1, 256))(x2)
        x3 = Reshape((16, 16, 1, 256))(x3)
        x4 = Reshape((16, 16, 1, 256))(x4)

        x = Concatenate(axis=3)([x0, x1, x2, x3, x4])

        x = Conv3D(256, (1, 1, 5), activation='relu', use_bias=False, kernel_initializer='he_normal')(x)
        x = Reshape((16, 16, 256))(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(x)
        x = BatchNormalization(name='concat_projection_BN', epsilon=1e-5)(x)
        x = Activation('relu')(x)
        x = Dropout(0.1)(x)

        # DeepLab v.3+ decoder
        # Feature projection

        x = BilinearUpsampling(output_size=(int(np.ceil(input_shape[0] / 4)),
                                            int(np.ceil(input_shape[1] / 4))), l_name='up2')(x)
        dec_skip1 = Conv2D(48, (1, 1), padding='same',
                           use_bias=False, name='feature_projection0')(skip1)
        dec_skip1 = BatchNormalization(
            name='feature_projection0_BN', epsilon=1e-5)(dec_skip1)
        dec_skip1 = Activation('relu')(dec_skip1)
        x = Concatenate()([x, dec_skip1])
        x = SepConv_BN(x, 256, 'decoder_conv0',
                       depth_activation=True, epsilon=1e-5)
        x = SepConv_BN(x, 256, 'decoder_conv1',
                       depth_activation=True, epsilon=1e-5)

        conv8 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x)
        conv8 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
        conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
        x = Conv2D(1, 1, activation='sigmoid')(conv9)
        x1 = BilinearUpsampling(output_size=(input_shape[0], input_shape[1]), l_name='x1')(x)

        if input_tensor is not None:
            inputs = get_source_inputs(input_tensor)
        else:
            inputs = img_input

        model = Model(inputs, x1, name='deeplabv3')
        return model
