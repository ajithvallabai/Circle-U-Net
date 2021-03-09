import tensorflow as tf
# tf.config.experimental_run_functions_eagerly(True)
# tf.enable_eager_execution()
# print(tf.executing_eagerly())

from tensorflow.keras import backend as K
smooth = 1.
#
# def tversky_loss(y_true, y_pred):
#     alpha = 0.5
#     beta = 0.5
#
#     ones = tf.ones(tf.shape(y_true))
#     p0 = y_pred  # proba that voxels are class i
#     p1 = ones - y_pred  # proba that voxels are not class i
#     g0 = y_true
#     g1 = ones - y_true
#
#     num = tf.sum(p0 * g0, (0, 1, 2, 3))
#     den = num + alpha * tf.sum(p0 * g1, (0, 1, 2, 3)) + beta * tf.sum(p1 * g0, (0, 1, 2, 3))
#
#     T =tf.sum(num / den)  # when summing over classes, T has dynamic range [0 Ncl]
#
#     Ncl = tf.cast(tf.shape(y_true)[-1], 'float32')
#     return Ncl - T
#
#
# def dice_coef(y_true, y_pred):
#     y_true_f = tf.flatten(y_true)
#     y_pred_f = tf.flatten(y_pred)
#     intersection = tf.sum(y_true_f * y_pred_f)
#     return (2. * intersection + smooth) / (K.sum(y_true_f * y_true_f) + K.sum(y_pred_f * y_pred_f) + smooth)
#
#
# def dice_coef_loss(y_true, y_pred):
#     return 1. - dice_coef(y_true, y_pred)

def weighted_dirichlet_loss(weights):

    class_weights = K.constant(weights)

    def loss(y_true, y_pred):
        pixel_weights = K.gather(class_weights, K.argmax(y_true, axis=-1))
        dist = tf.distributions.Dirichlet(1000*y_pred+K.epsilon())
        error = -dist.log_prob(y_true)
        loss = tf.reduce_sum(pixel_weights*error)

        return loss

    return loss

def tversky_loss(y_true, y_pred):
    alpha = 0.5
    beta = 0.5

    ones = K.ones(K.shape(y_true))
    p0 = y_pred  # proba that voxels are class i
    p1 = ones - y_pred  # proba that voxels are not class i
    g0 = y_true
    g1 = ones - y_true

    num = K.sum(p0 * g0, (0, 1, 2, 3))
    den = num + alpha * K.sum(p0 * g1, (0, 1, 2, 3)) + beta * K.sum(p1 * g0, (0, 1, 2, 3))

    T = K.sum(num / den)  # when summing over classes, T has dynamic range [0 Ncl]

    Ncl = K.cast(K.shape(y_true)[-1], 'float32')
    return Ncl - T


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f * y_true_f) + K.sum(y_pred_f * y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1. - dice_coef(y_true, y_pred)

# IOU calculation
def iou(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + smooth)


def iou_thresholded(y_true, y_pred, threshold=0.5, smooth=1.):
    y_pred = threshold_binarize(y_pred, threshold)
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + smooth)
