# Copyright 2021 The AI-Medic\Cell-Classification Authors. All Rights Reserved.
# License stuff will be written here later...

"""UNet Model implemented from scratch using tensorflow.keras

UNet is an architecture that has many use cases in image segmentation tasks.

It supports various image sizes starting from 32*32

In the architecture; an attention gate is also used and can be applied in the model
by the user (by setting attention=True when passing parameters)

In the table below you can see multiple performances resulted from different configurations
of UNet:

+--------------+---------------------+---------------------+--------------+---------------+
| Model Name   |   DSB-2018 Val Dice |   DSB-2018 Val Loss | Params (M)   | Attention     |
+==============+=====================+=====================+==============+===============+
| Unet         |               88.65 |              0.1438 | 31,032,837   | None          |
+--------------+---------------------+---------------------+--------------+---------------+
| Unet         |               90.1  |              0.18   | 39,050,081   | AttentionGate |
+--------------+---------------------+---------------------+--------------+---------------+

* DSB-2018 : DATA-SCIENCE-BOWL-2018

Reference:
  - [Github name](https://github.com/petrosgk/Kaggle-Carvana-Image-Masking-Challenge/tree/a4150d2d68b73ea5682334b976707a5e21fa043e/model)
"""

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, UpSampling2D, concatenate, MaxPooling2D
from utils.attention import AttnGatingBlock as attention
from params import get_args


class UNet:
    """
       UNet class
       Instantiates the UNet architecture.

       Reference:
             - [Github name](https://github.com/petrosgk/Kaggle-Carvana-Image-Masking-Challenge/tree/a4150d2d68b73ea5682334b976707a5e21fa043e/model)

       For image segmentation use cases, see
           [this page for detailed examples](
             https://keras.io/examples/vision/oxford_pets_image_segmentation/)
    """

    def __init__(self, input_shape=(256, 256, 3), **kwargs):
        """
        Parameters
          ----------
        input_shape: shape tuple, in "channels_last" format;
            it should have exactly 3 inputs channels, and width and
            height should be no smaller than 32.
            E.g. `(256, 256, 3)` would be one valid value. Default to `None`.
        """
        self.input_shape = input_shape

    def get_model(self) -> Model:
        """
        This method returns a Keras image segmentation model.

        Returns
        -------
        A `Tensorflow.keras.Model` instance.
        """

        def conv_layers(filters):
            def func(prev_layer):
                conv = Conv2D(filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(prev_layer)
                return Conv2D(filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv)

            return func

        def up_conv_layer(filters):
            def func(skip_layer, prev_layer):
                up = UpSampling2D(size=(2, 2))(prev_layer)
                up = Conv2D(filters, 2, activation='relu', padding='same', kernel_initializer='he_normal')(up)
                return concatenate([skip_layer, up], axis=3)

            return func

        inputs = Input(self.input_shape)

        conv1 = conv_layers(64)(inputs)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        conv2 = conv_layers(128)(pool1)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        conv3 = conv_layers(256)(pool2)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        conv4 = conv_layers(512)(pool3)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        conv5 = conv_layers(1024)(pool4)

        args = get_args('unet')
        if args.attention:
            attention4 = attention(conv4, conv5, 1024)
            attention3 = attention(conv3, conv4, 512)
            attention2 = attention(conv2, conv3, 256)
            attention1 = attention(conv1, conv2, 128)

            conv4 = attention4
            conv3 = attention3
            conv2 = attention2
            conv1 = attention1

        merge6 = up_conv_layer(512)(conv4, conv5)
        conv6 = conv_layers(512)(merge6)
        merge7 = up_conv_layer(256)(conv3, conv6)
        conv7 = conv_layers(256)(merge7)
        merge8 = up_conv_layer(128)(conv2, conv7)
        conv8 = conv_layers(128)(merge8)
        merge9 = up_conv_layer(64)(conv1, conv8)
        conv9 = conv_layers(64)(merge9)

        conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

        return Model(inputs=inputs, outputs=conv10)
