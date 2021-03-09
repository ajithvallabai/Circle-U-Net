from tensorflow.keras.layers import Input, Add, Dropout, Permute, add, concatenate, UpSampling2D
from tensorflow.keras.layers import Convolution2D, ZeroPadding2D, MaxPooling2D, Cropping2D, Conv2D, BatchNormalization
from tensorflow.compat.v1.layers import conv2d_transpose
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
import  tensorflow as tf

def create_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))


def create_biases(size):
    return tf.Variable(tf.constant(0.05, shape=[size]))


def conv_layer(stride,input, num_input_channels, conv_filter_size, num_filters, padding='SAME', relu=True):
    weights = create_weights(shape=[conv_filter_size, conv_filter_size, num_input_channels, num_filters])
    biases = create_biases(num_filters)
    layer = tf.nn.conv2d(input=input, filter=weights, strides=[1, stride, stride, 1], padding=padding)
    layer += biases
    if relu:
        layer = tf.nn.relu(layer)
    return layer


def pool_layer(input, padding='SAME'):
    return tf.nn.max_pool(value=input,
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding=padding)

#batch_size=5
def un_conv(stride,input,num_input_channels, conv_filter_size, num_filters, feature_map_size_W,feature_map_size_H, train=True, padding='SAME',
            relu=True):
    weights = create_weights(shape=[conv_filter_size, conv_filter_size, num_filters, num_input_channels])
    biases = create_biases(num_filters)
    batch_size = 5
    if train:
        batch_size_0 = batch_size
    else:
        batch_size_0 = 1
    layer = tf.nn.conv2d_transpose(value=input, filter=weights,
                                   output_shape=[batch_size_0, feature_map_size_W, feature_map_size_H, num_filters],
                                   strides=[1, stride, stride, 1],
                                   padding=padding)
    layer += biases
    if relu:
        layer = tf.nn.relu(layer)
    return layer


def create_unet(input, train=True, img_size_W=256, img_size_H=256, classes=24, batch_size=5):
    # strieds, inputlayer, channels, kernel_size, filters
    conv1 = conv_layer(1, input, 3, 3, 64)  # 360 x 480 x 64
    conv2 = conv_layer(1, conv1, 64, 3, 64)  # 360 x 480 x 64
    Dense0 = tf.concat([conv1, conv2], axis=-1)  # 360 x 480 x 128
    squeeze_conv1_1 = conv_layer(2, Dense0, 128, 1, 32)  # 180 x 240 x 32
    expand_conv1x1_1 = conv_layer(1, squeeze_conv1_1, 32, 1, 64, relu=False)  # 180 x 240 x 64
    expand_conv3x3_1 = conv_layer(1, squeeze_conv1_1, 32, 3, 64, relu=False)  # 180 x 240 x 64
    concat1_1 = tf.concat([expand_conv1x1_1, expand_conv3x3_1], axis=-1)  # 180 x 240 x 128
    squeeze_conv1_2 = conv_layer(1, concat1_1, 128, 1, 32)  # 180 x 240 x 32
    expand_conv1x1_2 = conv_layer(1, squeeze_conv1_2, 32, 1, 64, relu=False)  # 180 x 240 x 64
    expand_conv3x3_2 = conv_layer(1, expand_conv1x1_2, 32, 3, 64, relu=False)  # 180 x 240 x 64
    concat1_2 = tf.concat([expand_conv1x1_2, expand_conv3x3_2], axis=-1)  # 180 x 240 x 128

    Dense_1 = tf.concat([concat1_1, concat1_2], axis=-1)  # 180 x 240 x 256
    Dense_1 = tf.nn.relu(Dense_1)
    squeeze_conv2_1 = conv_layer(2, Dense_1, 256, 1, 48)  # 90 x 120 x 48
    expand_conv1x1_3 = conv_layer(1, squeeze_conv2_1, 48, 1, 128, relu=False)  # 90 x 120 x 128
    expand_conv3x3_3 = conv_layer(1, squeeze_conv2_1, 48, 3, 128, relu=False)  # 90 x 120 x 128
    concat2_1 = tf.concat([expand_conv1x1_3, expand_conv3x3_3], axis=-1)  # 90 x 120 x 256
    squeeze_conv2_2 = conv_layer(1, concat2_1, 256, 1, 48)  # 90 x 120 x 48
    expand_conv1x1_4 = conv_layer(1, squeeze_conv2_2, 48, 1, 128, relu=False)  # 90 x 120 x 128
    expand_conv3x3_4 = conv_layer(1, squeeze_conv2_2, 48, 3, 128, relu=False)  # 90 x 120 x 128
    concat2_2 = tf.concat([expand_conv1x1_4, expand_conv3x3_4], axis=-1)  # 90 x 120 x 256

    Dense_3 = tf.concat([concat2_1, concat2_2], axis=-1)  # 90 x 120 x 512
    squeeze_conv3_1 = conv_layer(2, Dense_3, 512, 1, 64)  # 45 x 60 x 64
    expand_conv1x1_5 = conv_layer(1, squeeze_conv3_1, 64, 1, 256, relu=False)  # #45 x 60 x 256
    expand_conv3x3_5 = conv_layer(1, squeeze_conv3_1, 64, 3, 256, relu=False)  # #45 x 60 x 256
    concat3_1 = tf.concat([expand_conv1x1_5, expand_conv3x3_5], axis=-1)  # #45 x 60 x 512
    squeeze_conv2_2 = conv_layer(1, concat3_1, 512, 1, 64)  # #45 x 60 x 64
    expand_conv1x1_4 = conv_layer(1, squeeze_conv2_2, 64, 1, 256, relu=False)  # #45 x 60 x 256
    expand_conv3x3_4 = conv_layer(1, squeeze_conv2_2, 64, 3, 256, relu=False)  ##45 x 60 x 256
    concat3_2 = tf.concat([expand_conv1x1_4, expand_conv3x3_4], axis=-1)  # 45 x 60 x 512

    Dense_5 = tf.concat([concat3_1, concat3_2], axis=-1)  # 45 x 60 x 1024
    squeeze_conv3_1 = conv_layer(2, Dense_5, 1024, 1, 80)  # 23 x 30 x 80
    expand_conv1x1_5 = conv_layer(1, squeeze_conv3_1, 80, 1, 512, relu=False)  # 23 x 30 x 512
    expand_conv3x3_5 = conv_layer(1, squeeze_conv3_1, 80, 3, 512, relu=False)  # 23 x 30 x 512
    concat4_1 = tf.concat([expand_conv1x1_5, expand_conv3x3_5], axis=-1)  # 23 x 30 x 1024
    squeeze_conv2_2 = conv_layer(1, concat4_1, 1024, 1, 80)  # 23 x 30 x 80
    expand_conv1x1_4 = conv_layer(1, squeeze_conv2_2, 80, 1, 512, relu=False)  # 23 x 30 x 512
    expand_conv3x3_4 = conv_layer(1, squeeze_conv2_2, 80, 3, 512, relu=False)  # 23 x 30 x 512
    concat4_2 = tf.concat([expand_conv1x1_4, expand_conv3x3_4], axis=-1)  # 23 x 30 x 1024

    Dense_6 = tf.concat([concat4_1, concat4_2], axis=-1)  # 23 x 30 x 2048
    # 1st up fire module
    # stride,input,num_input_channels, conv_filter_size, num_filters, feature_map_size_W,feature_map_size_H, train=True,
    s1 = un_conv(2, Dense_6, 2048, 1, 80, img_size_W // 8, img_size_H // 8, train)  # (45,60, 80
    e1x1_1 = un_conv(1, s1, 80, 1, 256, img_size_W // 8, img_size_H // 8, train)  # 45 x 60 x 256
    e3x3_1 = un_conv(1, s1, 80, 2, 256, img_size_W // 8, img_size_H // 8, train)  # 45 x 60 x 256
    c1 = tf.concat([e1x1_1, e3x3_1], axis=-1)  # 45 x 60 x 512
    merge11 = tf.concat(values=[concat3_2, c1], axis=-1)  # 45 x 60 x 1024
    squeeze_upconv0_1 = conv_layer(1, merge11, 1024, 1, 64)  # 45 x 60 x 64
    expand_upconv1x1_1 = conv_layer(1, squeeze_upconv0_1, 64, 1, 256, relu=False)  # 45 x 60 x 256
    expand_upconv3x3_1 = conv_layer(1, squeeze_upconv0_1, 64, 3, 256, relu=False)  # 45 x 60 x 256
    upconcat0_1 = tf.concat([expand_upconv1x1_1, expand_upconv3x3_1], axis=-1)  # 45 x 60 x 512
    squeeze_upconv0_2 = conv_layer(1, upconcat0_1, 512, 1, 64)  # 45 x 60 x 64
    expand_upconv1x1_2 = conv_layer(1, squeeze_upconv0_2, 64, 1, 256, relu=False)  # 45 x 60 x 256
    expand_upconv3x3_2 = conv_layer(1, squeeze_upconv0_2, 64, 3, 256, relu=False)  # 45 x 60 x 256
    upconcat0_2 = tf.concat([expand_upconv1x1_2, expand_upconv3x3_2], axis=-1)  # 45 x 60 x 512

    s2 = un_conv(2, upconcat0_2, 512, 1, 64, img_size_W // 4, img_size_H // 4, train)  # (8x80x80x64)
    e1x1_2 = un_conv(1, s2, 64, 1, 128, img_size_W // 4, img_size_H // 4, train)  # (8x80x80x128)
    e3x3_2 = un_conv(1, s2, 64, 2, 128, img_size_W // 4, img_size_H // 4, train)  # (8x80x80x128)
    c2 = tf.concat([e1x1_2, e3x3_2], axis=-1)  # 8x80x80x256
    merge14 = tf.concat([concat2_2, c2], axis=-1)  # 80x80x512
    squeeze_upconv1_1 = conv_layer(1, merge14, 512, 1, 48)  # 8x80x80x48
    expand_upconv1x1_3 = conv_layer(1, squeeze_upconv1_1, 48, 1, 128, relu=False)  # 8x80x80x128
    expand_upconv3x3_3 = conv_layer(1, squeeze_upconv1_1, 48, 3, 128, relu=False)  # 8x80x80x128
    upconcat1_1 = tf.concat([expand_upconv1x1_3, expand_upconv3x3_3], axis=-1)  # 8x80x80x256
    upconcat1_1 = tf.nn.relu(upconcat1_1)
    squeeze_upconv3_1 = conv_layer(1, upconcat1_1, 256, 1, 48)  # 8x80x80x64
    expand_upconv1x1_4 = conv_layer(1, squeeze_upconv3_1, 48, 1, 128, relu=False)  # 8x80x80x128
    expand_upconv3x3_4 = conv_layer(1, squeeze_upconv3_1, 48, 3, 128, relu=False)  # 8x80x80x128
    upconcat1_2 = tf.concat([expand_upconv1x1_4, expand_upconv3x3_4], axis=-1)  # 8x80x80x256
    upconcat1_2 = tf.nn.relu(upconcat1_2)

    s3 = un_conv(2, upconcat1_2, 256, 1, 48, img_size_W // 2, img_size_H // 2, train)  # (8x160x160x48)
    e1x1_3 = un_conv(1, s3, 48, 1, 64, img_size_W // 2, img_size_H // 2, train)  # (8x160x160x64)
    e3x3_3 = un_conv(1, s3, 48, 2, 64, img_size_W // 2, img_size_H // 2, train)  # (8x160x160x64)
    c3 = tf.concat([e1x1_3, e3x3_3], axis=-1)  # 8x160x160x128
    merge17 = tf.concat([c3, concat1_2], axis=-1)  # 160x160x256
    squeeze_upconv2_1 = conv_layer(1, merge17, 256, 1, 32)  # 8x320x320x32
    expand_upconv1x1_5 = conv_layer(1, squeeze_upconv2_1, 32, 1, 64, relu=False)  # 8x160x1600x64
    expand_upconv3x3_5 = conv_layer(1, squeeze_upconv2_1, 32, 3, 64, relu=False)  # 8x160x160x64
    upconcat2_1 = tf.concat([expand_upconv1x1_5, expand_upconv3x3_5], axis=-1)  # 8x160x160x128
    squeeze_upconv2_2 = conv_layer(1, upconcat2_1, 128, 1, 32)  # 8x320x320x32
    expand_upconv1x1_6 = conv_layer(1, squeeze_upconv2_2, 32, 1, 64, relu=False)  # 8x160x160x64
    expand_upconv3x3_6 = conv_layer(1, squeeze_upconv2_2, 32, 3, 64, relu=False)  # 8x160x160x64
    upconcat2_2 = tf.concat([expand_upconv1x1_6, expand_upconv3x3_6], axis=-1)  # 8x160x160x128

    conv20 = un_conv(2, upconcat2_2, 128, 2, 64, img_size_W, img_size_H, train)  # 320x320x64
    merge20 = tf.concat([conv20, conv2], axis=-1)  # 320x320x128
    conv21 = conv_layer(1, merge20, 128, 3, 64)  # 320x320x64
    conv22 = conv_layer(1, conv21, 64, 3, 64)  # 320x320x64
    conv23 = conv_layer(1, conv22, 64, 1, classes)  # 320x320x9
    return conv23


def UNet(batch_size, classes, n_filters=16, bn=True, dilation_rate=1):
    '''Validation Image data generator
        Inputs:
            n_filters - base convolution filters
            bn - flag to set batch normalization
            dilation_rate - convolution dilation rate
        Output: Unet keras Model
    '''
    # Define input batch shape
    batch_shape = (256, 256, 3)
    inputs = Input(batch_shape=(5, 256, 256, 3))
    print(inputs)
    conv23 = create_unet(inputs, train=True, img_size_W=256, img_size_H=256, batch_size=batch_size, classes=classes)
    #model = Model(inputs=inputs, outputs=conv23)
    return model



if __name__ == "__main__":
    batch_size = 5
    classes = 24
    model = UNet(batch_size,classes)

    #print(model_summary())
    #print(model.summary())