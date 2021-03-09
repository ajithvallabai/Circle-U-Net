from tensorflow.keras.layers import Input, Add, Dropout, Permute, add, concatenate, UpSampling2D
from tensorflow.keras.layers import Convolution2D, ZeroPadding2D, MaxPooling2D, Cropping2D, Conv2D, BatchNormalization
from tensorflow.compat.v1.layers import conv2d_transpose
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

def UNet(n_filters=16, bn=True, dilation_rate=1):
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

    conv1 = Conv2D(n_filters * 1, (3, 3), activation='relu', padding='same', dilation_rate=dilation_rate)(inputs)
    if bn:
        conv1 = BatchNormalization()(conv1)

    conv1 = Conv2D(n_filters * 1, (3, 3), activation='relu', padding='same', dilation_rate=dilation_rate)(conv1)
    if bn:
        conv1 = BatchNormalization()(conv1)

    pool1 = MaxPooling2D(pool_size=(2, 2), data_format='channels_last')(conv1)

    conv2 = Conv2D(n_filters * 2, (3, 3), activation='relu', padding='same', dilation_rate=dilation_rate)(pool1)
    if bn:
        conv2 = BatchNormalization()(conv2)

    conv2 = Conv2D(n_filters * 2, (3, 3), activation='relu', padding='same', dilation_rate=dilation_rate)(conv2)
    if bn:
        conv2 = BatchNormalization()(conv2)

    pool2 = MaxPooling2D(pool_size=(2, 2), data_format='channels_last')(conv2)

    conv3 = Conv2D(n_filters * 4, (3, 3), activation='relu', padding='same', dilation_rate=dilation_rate)(pool2)
    if bn:
        conv3 = BatchNormalization()(conv3)

    conv3 = Conv2D(n_filters * 4, (3, 3), activation='relu', padding='same', dilation_rate=dilation_rate)(conv3)
    if bn:
        conv3 = BatchNormalization()(conv3)

    pool3 = MaxPooling2D(pool_size=(2, 2), data_format='channels_last')(conv3)

    conv4 = Conv2D(n_filters * 8, (3, 3), activation='relu', padding='same', dilation_rate=dilation_rate)(pool3)
    if bn:
        conv4 = BatchNormalization()(conv4)

    conv4 = Conv2D(n_filters * 8, (3, 3), activation='relu', padding='same', dilation_rate=dilation_rate)(conv4)
    if bn:
        conv4 = BatchNormalization()(conv4)

    pool4 = MaxPooling2D(pool_size=(2, 2), data_format='channels_last')(conv4)

    conv5 = Conv2D(n_filters * 16, (3, 3), activation='relu', padding='same', dilation_rate=dilation_rate)(pool4)
    if bn:
        conv5 = BatchNormalization()(conv5)

    conv5 = Conv2D(n_filters * 16, (3, 3), activation='relu', padding='same', dilation_rate=dilation_rate)(conv5)
    if bn:
        conv5 = BatchNormalization()(conv5)

    up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=3)

    conv6 = Conv2D(n_filters * 8, (3, 3), activation='relu', padding='same', dilation_rate=dilation_rate)(up6)
    if bn:
        conv6 = BatchNormalization()(conv6)

    conv6 = Conv2D(n_filters * 8, (3, 3), activation='relu', padding='same', dilation_rate=dilation_rate)(conv6)
    if bn:
        conv6 = BatchNormalization()(conv6)

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=3)

    conv7 = Conv2D(n_filters * 4, (3, 3), activation='relu', padding='same', dilation_rate=dilation_rate)(up7)
    if bn:
        conv7 = BatchNormalization()(conv7)

    conv7 = Conv2D(n_filters * 4, (3, 3), activation='relu', padding='same', dilation_rate=dilation_rate)(conv7)
    if bn:
        conv7 = BatchNormalization()(conv7)

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=3)

    conv8 = Conv2D(n_filters * 2, (3, 3), activation='relu', padding='same', dilation_rate=dilation_rate)(up8)
    if bn:
        conv8 = BatchNormalization()(conv8)

    conv8 = Conv2D(n_filters * 2, (3, 3), activation='relu', padding='same', dilation_rate=dilation_rate)(conv8)
    if bn:
        conv8 = BatchNormalization()(conv8)

    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=3)

    conv9 = Conv2D(n_filters * 1, (3, 3), activation='relu', padding='same', dilation_rate=dilation_rate)(up9)
    if bn:
        conv9 = BatchNormalization()(conv9)

    conv9 = Conv2D(n_filters * 1, (3, 3), activation='relu', padding='same', dilation_rate=dilation_rate)(conv9)
    if bn:
        conv9 = BatchNormalization()(conv9)

    conv10 = Conv2D(24, (1, 1), activation='softmax', padding='same', dilation_rate=dilation_rate)(conv9)

    model = Model(inputs=inputs, outputs=conv10)

    return model