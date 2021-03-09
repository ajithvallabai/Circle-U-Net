"""
New model : SqueezeUnet + Attenation

"""

from tensorflow.keras.layers import Input, Activation, Add, Dropout, Permute, add, concatenate, UpSampling2D, Conv2DTranspose
from tensorflow.keras.layers import Convolution2D, ZeroPadding2D, MaxPooling2D, Cropping2D, Conv2D, BatchNormalization, \
    add, multiply
from tensorflow.compat.v1.layers import conv2d_transpose
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

# Conv2DTranspose()

# improve the model by writing proper batch normalization

## Reference : https://github.com/theislab/LODE/blob/3526cf6d2a2afb5907c5a1085d519170dd816a02/feature_segmentation/models/advanced_unets/models.py
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

def UNet(n_filters=16, bn=True, dilation_rate=1,batch_size=5,classes= 24):
    batch_shape = (256, 256, 3)
    inputs = Input(batch_shape=(5, 256, 256, 3))
    #print(inputs)

    conv1 = Conv2D(64, (3, 3), strides=(1, 1), activation='relu', padding='same', dilation_rate=dilation_rate)(inputs)
    if bn:
        conv1 = BatchNormalization()(conv1)

    conv2 = Conv2D(64, (3, 3), strides=(1, 1), activation='relu', padding='same', dilation_rate=dilation_rate)(conv1)
    if bn:
        conv1 = BatchNormalization()(conv2)

    Dense0 = concatenate([conv1, conv2], axis=-1)
    # filters, kernel_size, strides
    squeeze_conv1_1 = Conv2D(32, (1, 1), strides=(2, 2), activation='relu', padding='same', dilation_rate=dilation_rate)(Dense0)
    expand_conv1x1_1 = Conv2D(64, (1, 1), strides=(1, 1), padding='same', dilation_rate=dilation_rate)(squeeze_conv1_1)
    expand_conv3x3_1 = Conv2D(64, (3, 3), strides=(1, 1), padding='same', dilation_rate=dilation_rate)(squeeze_conv1_1)
    concat1_1 = concatenate([expand_conv1x1_1, expand_conv3x3_1], axis=-1)

    squeeze_conv1_2 = Conv2D(32, (1, 1), strides=(1, 1), activation='relu', padding='same', dilation_rate=dilation_rate)(concat1_1)
    expand_conv1x1_2 = Conv2D(64, (1, 1), strides=(1, 1), padding='same', dilation_rate=dilation_rate)(squeeze_conv1_2)
    expand_conv3x3_2 = Conv2D(64, (3, 3), strides=(1, 1), padding='same', dilation_rate=dilation_rate)(expand_conv1x1_2)
    concat1_2 = concatenate([expand_conv1x1_2, expand_conv3x3_2], axis=-1)

    Dense_1 =  concatenate([concat1_1, concat1_2], axis=-1)
    Dense_1 = Activation('relu')(Dense_1)
    squeeze_conv2_1 = Conv2D(48, (1, 1), strides=(2, 2),activation='relu', padding='same', dilation_rate=dilation_rate)(Dense_1)
    expand_conv1x1_3 = Conv2D(128, (1, 1), strides=(1, 1), padding='same', dilation_rate=dilation_rate)(squeeze_conv2_1)
    expand_conv3x3_3 = Conv2D(128, (3, 3), strides=(1, 1), padding='same', dilation_rate=dilation_rate)(squeeze_conv2_1)
    concat2_1 = concatenate([expand_conv1x1_3, expand_conv3x3_3], axis=-1)
    squeeze_conv2_2 = Conv2D(48, (1, 1), strides=(1, 1), activation='relu', padding='same', dilation_rate=dilation_rate)(concat2_1)
    expand_conv1x1_4 = Conv2D(128, (1, 1), strides=(1, 1), padding='same', dilation_rate=dilation_rate)(squeeze_conv2_2)
    expand_conv3x3_4 = Conv2D(128, (3, 3), strides=(1, 1), padding='same', dilation_rate=dilation_rate)(squeeze_conv2_2)
    concat2_2 = concatenate([expand_conv1x1_4, expand_conv3x3_4], axis=-1)
    Dense_3 = concatenate([concat2_1, concat2_2], axis=-1)

    #Dense_1 = Activation('relu')
    squeeze_conv3_1 = Conv2D(64, (1, 1), strides=(2, 2), activation='relu', padding='same', dilation_rate=dilation_rate)(Dense_3)
    expand_conv1x1_5 = Conv2D(256, (1, 1), strides=(1, 1), padding='same', dilation_rate=dilation_rate)(squeeze_conv3_1)
    expand_conv3x3_5 = Conv2D(256, (3, 3), strides=(1, 1), padding='same', dilation_rate=dilation_rate)(squeeze_conv3_1)
    concat3_1 = concatenate([expand_conv1x1_5 ,expand_conv3x3_5], axis=-1)
    squeeze_conv2_2 = Conv2D(64, (1, 1), strides=(1, 1), activation='relu', padding='same', dilation_rate=dilation_rate)(concat3_1)
    expand_conv1x1_4 = Conv2D(256, (1, 1), strides=(1, 1), padding='same', dilation_rate=dilation_rate)(squeeze_conv2_2)
    expand_conv3x3_4 = Conv2D(256, (3, 3), strides=(1, 1), padding='same', dilation_rate=dilation_rate)(squeeze_conv2_2)
    concat3_2 = concatenate([expand_conv1x1_4, expand_conv3x3_4], axis=-1)
    Dense_5 = concatenate([concat3_1, concat3_2], axis=-1)

    #Dense_1 = Activation('relu')
    squeeze_conv3_1 = Conv2D(80, (1, 1), strides=(2, 2), activation='relu', padding='same', dilation_rate=dilation_rate)(Dense_5)
    expand_conv1x1_5 = Conv2D(512, (1, 1), strides=(1, 1), padding='same', dilation_rate=dilation_rate)(squeeze_conv3_1)
    expand_conv3x3_5 = Conv2D(512, (3, 3), strides=(1, 1), padding='same', dilation_rate=dilation_rate)(squeeze_conv3_1)
    concat4_1 = concatenate([expand_conv1x1_5 ,expand_conv3x3_5], axis=-1)
    squeeze_conv2_2 = Conv2D(80, (1, 1), strides=(1, 1), activation='relu', padding='same', dilation_rate=dilation_rate)(concat4_1)
    expand_conv1x1_4 = Conv2D(512, (1, 1), strides=(1, 1), padding='same', dilation_rate=dilation_rate)(squeeze_conv2_2)
    expand_conv3x3_4 = Conv2D(512, (3, 3), strides=(1, 1), padding='same', dilation_rate=dilation_rate)(squeeze_conv2_2)
    concat4_2 = concatenate([expand_conv1x1_4, expand_conv3x3_4], axis=-1)
    Dense_6 = concatenate([concat4_1, concat4_2], axis=-1)

    # 1st up fire module
    s1 = Conv2DTranspose(80,(1,1),strides=(2,2), activation='relu', padding='same', dilation_rate=dilation_rate)(Dense_6)
    e1x1_1 = Conv2DTranspose(256,(1,1),strides=(1,1), activation='relu', padding='same', dilation_rate=dilation_rate)(s1)
    e3x3_1 = Conv2DTranspose(256,(1,1),strides=(1,1), activation='relu', padding='same', dilation_rate=dilation_rate)(s1)
    c1 = concatenate([e1x1_1, e3x3_1], axis=-1)
    attention_s1 = attention_block_2d(concat3_2,e3x3_1,256)
    merge11 = concatenate([attention_s1, c1], axis=-1)
    squeeze_upconv0_1 = Conv2D(64, (1, 1), strides=(1, 1), padding='same', dilation_rate=dilation_rate)(merge11)
    expand_upconv1x1_1 = Conv2D(256, (1, 1), strides=(1, 1), padding='same', dilation_rate=dilation_rate)(squeeze_upconv0_1)
    expand_upconv3x3_1 = Conv2D(256, (3, 3), strides=(1, 1), padding='same', dilation_rate=dilation_rate)(squeeze_upconv0_1)
    upconcat0_1 = concatenate([expand_upconv1x1_1, expand_upconv3x3_1], axis=-1)
    squeeze_upconv0_2 = Conv2D(64, (1, 1), strides=(1, 1), activation='relu', padding='same', dilation_rate=dilation_rate)(upconcat0_1)
    expand_upconv1x1_2 = Conv2D(256, (1, 1), strides=(1, 1), padding='same', dilation_rate=dilation_rate)(squeeze_upconv0_2)
    expand_upconv3x3_2 = Conv2D(256, (3, 3), strides=(1, 1), padding='same', dilation_rate=dilation_rate)(squeeze_upconv0_2)
    upconcat0_2 = concatenate([expand_upconv1x1_2, expand_upconv3x3_2], axis=-1)


    s2 = Conv2DTranspose(64, (1,1),strides=(2,2), padding='same', activation='relu',dilation_rate=dilation_rate)(upconcat0_2)
    e1x1_2 = Conv2DTranspose(128,(1,1),strides=(1,1), padding='same', activation='relu', dilation_rate=dilation_rate)(s2)
    e3x3_2 = Conv2DTranspose(128,(2,2),strides=(1,1), padding='same', activation='relu', dilation_rate=dilation_rate)(s2)
    c2 = concatenate([e1x1_2, e3x3_2], axis=-1)
    attention_s2 = attention_block_2d(concat2_2, e3x3_2, 128)
    merge14 = concatenate([attention_s2, c2], axis=-1)
    squeeze_upconv1_1 = Conv2D(48, (1, 1), strides=(1, 1), activation='relu', padding='same', dilation_rate=dilation_rate)(merge14)
    expand_upconv1x1_3 = Conv2D(128, (1, 1), strides=(1, 1), padding='same', dilation_rate=dilation_rate)(squeeze_upconv1_1)
    expand_upconv3x3_3 = Conv2D(128, (3, 3), strides=(1, 1), padding='same', dilation_rate=dilation_rate)(squeeze_upconv1_1)
    upconcat1_1 = concatenate([expand_upconv1x1_3, expand_upconv3x3_3], axis=-1)
    upconcat1_1 = Activation('relu')(upconcat1_1)
    squeeze_upconv3_1 = Conv2D(48, (1, 1), strides=(1, 1), activation='relu', padding='same', dilation_rate=dilation_rate)(upconcat1_1)
    expand_upconv1x1_4 = Conv2D(128, (1, 1), strides=(1, 1), padding='same', dilation_rate=dilation_rate)(squeeze_upconv3_1)
    expand_upconv3x3_4 = Conv2D(128, (3, 3), strides=(1, 1), padding='same', dilation_rate=dilation_rate)(squeeze_upconv3_1)
    upconcat1_2 = concatenate([expand_upconv1x1_4, expand_upconv3x3_4], axis=-1)
    upconcat1_2 = Activation('relu')(upconcat1_2)

    s3 = Conv2DTranspose(48, (1,1),strides=(2,2), activation='relu', padding='same', dilation_rate=dilation_rate)(upconcat1_2)
    e1x1_3 = Conv2DTranspose(64,(1,1),strides=(1,1), activation='relu', padding='same', dilation_rate=dilation_rate)(s3)
    e3x3_3 = Conv2DTranspose(64,(2,2),strides=(1,1), activation='relu', padding='same', dilation_rate=dilation_rate)(s3)
    c3 = concatenate([e1x1_3, e3x3_3], axis=-1)
    attention_s3 = attention_block_2d(concat1_2, e3x3_3, 64)
    merge17 = concatenate([c3, attention_s3], axis=-1)
    squeeze_upconv2_1 = Conv2D(32, (1, 1), strides=(1, 1), activation='relu', padding='same', dilation_rate=dilation_rate)(merge17)
    expand_upconv1x1_5 = Conv2D(64, (1, 1), strides=(1, 1), padding='same', dilation_rate=dilation_rate)(squeeze_upconv2_1)
    expand_upconv3x3_5 = Conv2D(64, (3, 3), strides=(1, 1), padding='same', dilation_rate=dilation_rate)(squeeze_upconv2_1)
    upconcat2_1 = concatenate([expand_upconv1x1_5, expand_upconv3x3_5], axis=-1)
    #upconcat1_1 = Activation('relu')
    squeeze_upconv2_2 = Conv2D(32, (1, 1), strides=(1, 1), activation='relu', padding='same', dilation_rate=dilation_rate)(upconcat2_1)
    expand_upconv1x1_6 = Conv2D(64, (1, 1), strides=(1, 1), padding='same', dilation_rate=dilation_rate)(squeeze_upconv2_2)
    expand_upconv3x3_6 = Conv2D(64, (3, 3), strides=(1, 1), padding='same', dilation_rate=dilation_rate)(squeeze_upconv2_2)
    upconcat2_2 = concatenate([expand_upconv1x1_6, expand_upconv3x3_6], axis=-1)
    #upconcat1_2 = Activation('relu')

    conv20 = Conv2DTranspose(64, (2, 2), strides=(2, 2), activation='relu', padding='same', dilation_rate=dilation_rate)(upconcat2_2)
    merge20 = concatenate([conv20, conv2], axis=-1)
    conv21 = Conv2D(64, (3, 3), strides=(1, 1), activation='relu', padding='same', dilation_rate=dilation_rate)(merge20)
    conv22 = Conv2D(64, (3, 3), strides=(1, 1), activation='relu', padding='same', dilation_rate=dilation_rate)(conv21)
    conv23 = Conv2D(classes , (1, 1), strides=(1, 1), activation='relu', padding='same', dilation_rate=dilation_rate)(conv22)

    model = Model(inputs=inputs, outputs=conv23)

    return model

if __name__ == "__main__":
    im = UNet(classes= 24)
















