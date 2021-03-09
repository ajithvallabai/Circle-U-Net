from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Activation, Conv2D
from tensorflow.keras.layers import MaxPooling2D, BatchNormalization
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import add


def conv3x3(x, out_filters, strides=(1, 1)):
    x = Conv2D(out_filters, 3, padding='same', strides=strides, use_bias=False, kernel_initializer='he_normal')(x)
    return x


def Conv2d_BN(x, nb_filter, kernel_size, strides=(1, 1), padding='same', use_activation=True):
    x = Conv2D(nb_filter, kernel_size, padding=padding, strides=strides, kernel_initializer='he_normal')(x)
    x = BatchNormalization(axis=3)(x)
    if use_activation:
        x = Activation('relu')(x)
        return x
    else:
        return x


def basic_Block(input, out_filters, strides=(1, 1), with_conv_shortcut=False):
    x = conv3x3(input, out_filters, strides)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    x = conv3x3(x, out_filters)
    x = BatchNormalization(axis=3)(x)

    if with_conv_shortcut:
        residual = Conv2D(out_filters, 1, strides=strides, use_bias=False, kernel_initializer='he_normal')(input)
        residual = BatchNormalization(axis=3)(residual)
        x = add([x, residual])
    else:
        x = add([x, input])

    x = Activation('relu')(x)
    return x


def bottleneck_Block(input, out_filters, strides=(1, 1), with_conv_shortcut=False):
    expansion = 4
    de_filters = int(out_filters / expansion)

    x = Conv2D(de_filters, 1, use_bias=False, kernel_initializer='he_normal')(input)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    x = Conv2D(de_filters, 3, strides=strides, padding='same', use_bias=False, kernel_initializer='he_normal')(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    x = Conv2D(out_filters, 1, use_bias=False, kernel_initializer='he_normal')(x)
    x = BatchNormalization(axis=3)(x)

    if with_conv_shortcut:
        residual = Conv2D(out_filters, 1, strides=strides, use_bias=False, kernel_initializer='he_normal')(input)
        residual = BatchNormalization(axis=3)(residual)
        x = add([x, residual])
    else:
        x = add([x, input])

    x = Activation('relu')(x)
    return x


def unet_resnet_101(height, width, channel, classes):
    input = Input(shape=(height, width, channel))

    conv1_1 = Conv2D(64, 7, strides=(2, 2), padding='same', use_bias=False, kernel_initializer='he_normal')(input)
    conv1_1 = BatchNormalization(axis=3)(conv1_1)
    conv1_1 = Activation('relu')(conv1_1)
    conv1_2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(conv1_1)

    # conv2_x  1/4
    conv2_1 = bottleneck_Block(conv1_2, 256, strides=(1, 1), with_conv_shortcut=True)
    conv2_2 = bottleneck_Block(conv2_1, 256)
    conv2_3 = bottleneck_Block(conv2_2, 256)

    # conv3_x  1/8
    conv3_1 = bottleneck_Block(conv2_3, 512, strides=(2, 2), with_conv_shortcut=True)
    conv3_2 = bottleneck_Block(conv3_1, 512)
    conv3_3 = bottleneck_Block(conv3_2, 512)
    conv3_4 = bottleneck_Block(conv3_3, 512)

    # conv4_x  1/16
    conv4_1 = bottleneck_Block(conv3_4, 1024, strides=(2, 2), with_conv_shortcut=True)
    conv4_2 = bottleneck_Block(conv4_1, 1024)
    conv4_3 = bottleneck_Block(conv4_2, 1024)
    conv4_4 = bottleneck_Block(conv4_3, 1024)
    conv4_5 = bottleneck_Block(conv4_4, 1024)
    conv4_6 = bottleneck_Block(conv4_5, 1024)
    conv4_7 = bottleneck_Block(conv4_6, 1024)
    conv4_8 = bottleneck_Block(conv4_7, 1024)
    conv4_9 = bottleneck_Block(conv4_8, 1024)
    conv4_10 = bottleneck_Block(conv4_9, 1024)
    conv4_11 = bottleneck_Block(conv4_10, 1024)
    conv4_12 = bottleneck_Block(conv4_11, 1024)
    conv4_13 = bottleneck_Block(conv4_12, 1024)
    conv4_14 = bottleneck_Block(conv4_13, 1024)
    conv4_15 = bottleneck_Block(conv4_14, 1024)
    conv4_16 = bottleneck_Block(conv4_15, 1024)
    conv4_17 = bottleneck_Block(conv4_16, 1024)
    conv4_18 = bottleneck_Block(conv4_17, 1024)
    conv4_19 = bottleneck_Block(conv4_18, 1024)
    conv4_20 = bottleneck_Block(conv4_19, 1024)
    conv4_21 = bottleneck_Block(conv4_20, 1024)
    conv4_22 = bottleneck_Block(conv4_21, 1024)
    conv4_23 = bottleneck_Block(conv4_22, 1024)

    # conv5_x  1/32
    conv5_1 = bottleneck_Block(conv4_23, 2048, strides=(2, 2), with_conv_shortcut=True)
    conv5_2 = bottleneck_Block(conv5_1, 2048)
    conv5_3 = bottleneck_Block(conv5_2, 2048)

    up6 = Conv2d_BN(UpSampling2D(size=(2, 2))(conv5_3), 1024, 2)
    merge6 = concatenate([conv4_23, up6], axis=3)
    conv6 = Conv2d_BN(merge6, 1024, 3)
    conv6 = Conv2d_BN(conv6, 1024, 3)

    up7 = Conv2d_BN(UpSampling2D(size=(2, 2))(conv6), 512, 2)
    merge7 = concatenate([conv3_4, up7], axis=3)
    conv7 = Conv2d_BN(merge7, 512, 3)
    conv7 = Conv2d_BN(conv7, 512, 3)

    up8 = Conv2d_BN(UpSampling2D(size=(2, 2))(conv7), 256, 2)
    merge8 = concatenate([conv2_3, up8], axis=3)
    conv8 = Conv2d_BN(merge8, 256, 3)
    conv8 = Conv2d_BN(conv8, 256, 3)

    up9 = Conv2d_BN(UpSampling2D(size=(2, 2))(conv8), 64, 2)
    merge9 = concatenate([conv1_1, up9], axis=3)
    conv9 = Conv2d_BN(merge9, 64, 3)
    conv9 = Conv2d_BN(conv9, 64, 3)

    up10 = Conv2d_BN(UpSampling2D(size=(2, 2))(conv9), 64, 2)
    conv10 = Conv2d_BN(up10, 64, 3)
    conv10 = Conv2d_BN(conv10, 64, 3)

    conv11 = Conv2d_BN(conv10, classes, 1, use_activation=None)
    activation = Activation('sigmoid', name='Classification')(conv11)

    model = Model(inputs=input, outputs=activation)
    return model


# from keras.utils import plot_model


# model = unet_resnet_101(height=512, width=512, channel=3, classes=1)
# print(model.summary())
# plot_model(model, to_file='unet_resnet101.png', show_shapes=True)#coding=utf-8


#from tensorflow.keras.models import *
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import *
# import tensorflow.keras.backend as K
# import tensorflow as tf
# import numpy as np

# def pool_block(inp, pool_factor):
#     h = K.int_shape(inp)[1]
#     w = K.int_shape(inp)[2]
#     pool_size = strides = [int(np.round( float(h) / pool_factor)), int(np.round( float(w)/ pool_factor))]
#     x = AveragePooling2D(pool_size, strides=strides, padding='same')(inp)
#     x = Conv2D(256, (1, 1), padding='same', activation='relu')(x)
#     x = BatchNormalization()(x)
#     x = Lambda(lambda x: tf.image.resize_bilinear(x, size=(int(x.shape[1])*strides[0], int(x.shape[2])*strides[1])))(x)
#     x = Conv2D(256, (1, 1), padding='same', activation='relu')(x)
#     return x

# def PSPNet(nClasses, input_width=384, input_height=384):
#     assert input_height % 192 == 0
#     assert input_width % 192 == 0
#     inputs = Input(shape=(input_height, input_width, 3))

#     x = (Conv2D(64, (3, 3), activation='relu', padding='same'))(inputs)
#     x = (BatchNormalization())(x)
#     x = (MaxPooling2D((2, 2)))(x)
#     f1 = x
#     # 192 x 192

#     x = (Conv2D(128, (3, 3), activation='relu', padding='same'))(x)
#     x = (BatchNormalization())(x)
#     x = (MaxPooling2D((2, 2)))(x)
#     f2 = x
#     # 96 x 96
#     x = (Conv2D(256, (3, 3), activation='relu', padding='same'))(x)
#     x = (BatchNormalization())(x)
#     x = (MaxPooling2D((2, 2)))(x)
#     f3 = x
#     # 48 x 48
#     x = (Conv2D(256, (3, 3), activation='relu', padding='same'))(x)
#     x = (BatchNormalization())(x)
#     x = (MaxPooling2D((2, 2)))(x)
#     f4 = x

#     # 24 x 24
#     o = f4
#     pool_factors = [1, 2, 3, 6]
#     pool_outs = [o]
#     for p in pool_factors:
#         pooled = pool_block(o, p)
#         pool_outs.append(pooled)

#     o = Concatenate(axis=3)(pool_outs)
#     o = Conv2D(256, (3, 3), activation='relu', padding='same')(o)
#     o = BatchNormalization()(o)

#     o = Lambda(lambda x: tf.image.resize_bilinear(x, size=(int(x.shape[1])*8, int(x.shape[2])*8)))(x)
#     o = Conv2D(nClasses, (1, 1), padding='same')(o)
#     o_shape = Model(inputs, o).output_shape
#     outputHeight = o_shape[1]
#     outputWidth = o_shape[2]
#     print(outputHeight)
#     print(outputWidth)
#     #o = (Reshape((outputHeight*outputWidth, nClasses)))(o)
#     o = (Activation('softmax'))(o)
#     model = Model(inputs, o)
#     # ss

#     return model

# if __name__ == "__main__":
#     im = PSPNet(nClasses= 32)
#     print(im.summary())
