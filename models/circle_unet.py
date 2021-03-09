from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Activation, Conv2D
from tensorflow.keras.layers import MaxPooling2D, BatchNormalization
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import add,multiply

# conv2_1 to conv3_4
# conv3_1 to conv 4_11
# conv 4_11 to conv 4_23
# conv 4_1 to conv 5_3 
# add attenations

def attention_block_2d(x, g, inter_channel, data_format='channels_last'):
    # theta_x(?,g_height,g_width,inter_channel)

    theta_x = Conv2D(inter_channel, [1, 1], strides=[1, 1], data_format=data_format)(x)
    # phi_g(?,g_height,g_width,inter_channel)
    phi_g = Conv2D(inter_channel, [1, 1], strides=[1, 1], data_format=data_format)(g)
    # f(?,g_height,g_width,inter_channel)
    f = Activation('relu')(add([theta_x, phi_g]))
    # psi_f(?,g_height,g_width,1)
    psi_f = Conv2D(1, [1, 1], strides=[1, 1], data_format=data_format)(f)
    rate = Activation('sigmoid')(psi_f)

    # rate(?,x_height,x_width)

    # att_x(?,x_height,x_width,x_channel)

    att_x = multiply([x, rate])
    #print(att_x.shape)
    return att_x


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

def circle_connect(layer1,layer2,stride,out_filters):
    residual = Conv2D(out_filters, 1, strides=stride, use_bias=False, kernel_initializer='he_normal')(layer1)
    residual = BatchNormalization(axis=3)(residual)
    x = add([layer2, residual])

    return x


def Circle_unet_resnet_101(height, width, channel, classes):
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
    print("conv 3_4 shape = " ,conv3_4.shape)
    # conv2_1 to conv3_4
    cc1 = circle_connect(conv2_1,conv3_4,(2,2),512)    
    # print("cc1 shape",cc1.shape)

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
    # conv3_1 to conv 4_11
    cc2 = circle_connect(conv3_1,conv4_11,(2,2),1024)    
    # print("cc2 shape",cc2.shape)


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
    # conv 4_11 to conv 4_23
    cc3 = circle_connect(conv4_11,conv4_23,(1,1),1024)    
    # print("cc3 shape",cc3.shape)



    # conv5_x  1/32
    conv5_1 = bottleneck_Block(conv4_23, 2048, strides=(2, 2), with_conv_shortcut=True)
    conv5_2 = bottleneck_Block(conv5_1, 2048)
    conv5_3 = bottleneck_Block(conv5_2, 2048)
    # conv 4_1 to conv 5_3 
    cc4 = circle_connect(conv4_1,conv5_3,(2,2),2048)    
    # print("cc4 shape",cc4.shape)

    up6 = Conv2d_BN(UpSampling2D(size=(2, 2))(conv5_3), 1024, 2)
    merge6 = concatenate([conv4_23, up6], axis=3)
    # up6 = Conv2d_BN(UpSampling2D(size=(2, 2))(conv5_3), 1024, 2)
    # attention_s6 = attention_block_2d(up6, conv4_23, 1024)
    # merge6 = concatenate([attention_s6, up6 ] , axis=3 )
    conv6 = Conv2d_BN(merge6, 1024, 3)
    conv6 = Conv2d_BN(conv6, 1024, 3)


    up7 = Conv2d_BN(UpSampling2D(size=(2, 2))(conv6), 512, 2)
    merge7 = concatenate([conv3_4, up7], axis=3)    
    # up7 = Conv2d_BN(UpSampling2D(size=(2, 2))(conv6), 512, 2)
    # attention_s7 = attention_block_2d(up7, conv3_4, 512)
    # merge7 = concatenate([attention_s7, up7], axis=3)
    conv7 = Conv2d_BN(merge7, 512, 3)
    conv7 = Conv2d_BN(conv7, 512, 3)

    up8 = Conv2d_BN(UpSampling2D(size=(2, 2))(conv7), 256, 2)
    merge8 = concatenate([conv2_3, up8], axis=3)
    # up8 = Conv2d_BN(UpSampling2D(size=(2, 2))(conv7), 256, 2)
    # attention_s8 = attention_block_2d(up8, conv2_3, 256)
    # merge8 = concatenate([attention_s8, up8], axis=3)
    conv8 = Conv2d_BN(merge8, 256, 3)
    conv8 = Conv2d_BN(conv8, 256, 3)

    up9 = Conv2d_BN(UpSampling2D(size=(2, 2))(conv8), 64, 2)
    merge9 = concatenate([conv1_1, up9], axis=3)
    # up9 = Conv2d_BN(UpSampling2D(size=(2, 2))(conv8), 64, 2)
    # attention_s9 = attention_block_2d(up9, conv1_1, 64)
    # merge9 = concatenate([attention_s9, up9], axis=3)
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
#print(model.summary())
# plot_model(model, to_file='unet_resnet101.png', show_shapes=True)#coding=utf-8

