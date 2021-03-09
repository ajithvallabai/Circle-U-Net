"""
Used to evaluate model
python evaluate.py -d "camvid" -idir "dataset/camvid/data/" -mt "squeeze_unet_keras" -m "camvid_model_5_epochs.h5" -ht 256 -w 256
"""

import numpy as np
import os
import sys
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import argparse
from dataloader.look_in_person import read_images, parse_code
from dataloader.look_in_person import TrainAugmentGenerator, ValAugmentGenerator
from loss.loss import tversky_loss, dice_coef, dice_coef_loss
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import matplotlib.pyplot as plt

# helper function for data visualization
def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()


# helper function for data visualization
def denormalize(x):
    """Scale image to range 0..1 for correct plot"""
    x_max = np.percentile(x, 98)
    x_min = np.percentile(x, 2)
    x = (x - x_min) / (x_max - x_min)
    x = x.clip(0, 1)
    return x

def seglayers2mask(image, output, random_colors=True):
    # infer the total number of classes along with the spatial dimensions
    # of the mask image via the shape of the output array
    (height, width, numClasses) = output.shape[1:4]
    # print("[INFO] Number of classes: {:d}".format(numClasses))

    # our output class ID map will be num_classes x height x width in
    # size, so we take the argmax to find the class label with the
    # largest probability for each and every (x, y)-coordinate in the
    # image
    classMap = np.argmax(output[0], axis=-1)

    # given the class ID map, we can map each of the class IDs to its
    # corresponding color
    if random_colors:
        np.random.seed(4)
        COLORS = np.random.randint(0, 255, size=(numClasses - 1, 3), dtype="uint8")
        COLORS = np.vstack([COLORS, [0, 0, 0]]).astype("uint8")
    else:
        COLORS = open('colors.txt').read().strip().split("\n")
        COLORS = [np.array(c.split(",")).astype("int") for c in COLORS]
        COLORS = np.array(COLORS, dtype="uint8")

    mask = COLORS[classMap]

    # resize the mask and class map such that its dimensions match the
    # original size of the input image (we're not using the class map
    # here for anything else but this is how you would resize it just in
    # case you wanted to extract specific pixels/classes)
    mask = cv2.resize(mask, (image.shape[2], image.shape[1]), interpolation=cv2.INTER_NEAREST)
    # classMap = cv2.resize(classMap, (image.shape[2], image.shape[1]), interpolation=cv2.INTER_NEAREST)

    # perform a weighted combination of the input image with the mask to
    # form an output visualization
    rescaled_image = image[0] - np.min(image[0])
    rescaled_image /= np.max(rescaled_image)
    rescaled_image *= 255  # resaled image has color values in range 0..255
    output = ((0.4 * rescaled_image) + (0.6 * mask)).astype("uint8")

    return output

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset",required=True,
                    help="dataset")
    ap.add_argument("-idir", "--img_directory", required=True,
                    help="image directory")
    ap.add_argument("-mt", "--model_type", required=True,
                    help="model")
    ap.add_argument("-m", "--model_file", required=True,
                    help="model")
    ap.add_argument("-ht", "--output_height", required=False, default=256,
                    help="output height")
    ap.add_argument("-w", "--output_width", required=False, default=256,
                    help="output width")
    args = vars(ap.parse_args())
    print(args)

    # get as arguments
    dataset = args['dataset']
    img_dir = args['img_directory']
    DATA_PATH = args['img_directory']
    model_type = args['model_type']
    model_file = args['model_file']

    if dataset == "look_in_person" or dataset == "camvid":
        from dataloader.look_in_person import read_images, parse_code
        from dataloader.look_in_person import TrainAugmentGenerator, ValAugmentGenerator
    elif dataset == "camvid_full":
        from dataloader.camvid_full import read_images, parse_code
        from dataloader.camvid_full import TrainAugmentGenerator, ValAugmentGenerator

    x = tf.random.uniform([3, 3])
    print("Is there a GPU available: "),
    print(tf.test.is_gpu_available())
    print("Is the Tensor on GPU #0:  "),
    print(x.device.endswith('GPU:0'))
    print("Device name: {}".format((x.device)))
    print(tf.executing_eagerly())

    frame_tensors, masks_tensors, frames_list, masks_list = read_images(img_dir)

    # Make an iterator to extract images from the tensor dataset
    frame_batches = tf.compat.v1.data.make_one_shot_iterator(frame_tensors)  # outside of TF Eager, we would use make_one_shot_iterator
    mask_batches = tf.compat.v1.data.make_one_shot_iterator(masks_tensors)

    #generate_image_folder_structure(frame_tensors, masks_tensors, frames_list, masks_list)

    label_codes, label_names = zip(*[parse_code(l) for l in open(DATA_PATH +"label_color.txt")])
    label_codes, label_names = list(label_codes), list(label_names)
    #label_codes[:5], label_names[:5]
    code2id = {v: k for k, v in enumerate(label_codes)}
    id2code = {k: v for k, v in enumerate(label_codes)}
    name2id = {v: k for k, v in enumerate(label_names)}
    id2name = {k: v for k, v in enumerate(label_names)}
    #label_codes, label_names

    # print(id2code)
    # print(id2name)

    # Normalizing only frame images, since masks contain label info
    data_gen_args = dict(rescale=1. / 255)
    mask_gen_args = dict()

    # train_frames_datagen = ImageDataGenerator(**data_gen_args)
    # train_masks_datagen = ImageDataGenerator(**mask_gen_args)
    val_frames_datagen = ImageDataGenerator(**data_gen_args)
    val_masks_datagen = ImageDataGenerator(**mask_gen_args)

    # Seed defined for aligning images and their masks
    seed = 1

    if model_type == "tiny_unet":
        from models.small_unet import UNet
        model = UNet(n_filters = 32)
    elif model_type == "squeeze_unet_tf":
        from models.squeeze_unet_tf import UNet
        batch_size = 5
        classes = 24
        model = UNet(batch_size,classes)
    elif model_type == "squeeze_unet_keras":
        from models.squeeze_unet_keras import UNet
        batch_size = 5
        classes = 32
        model = UNet(batch_size=5,classes=classes)

    import segmentation_models as sm

    metrics_eval = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]
    model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=[tversky_loss, dice_coef, 'accuracy', \
                                                                              sm.metrics.IOUScore(threshold=0.5),
                                                                              sm.metrics.FScore(threshold=0.5)])
    model.summary()
    tb = TensorBoard(log_dir='logs', write_graph=True)
    mc = ModelCheckpoint(mode='max', filepath='camvid_model_5_epochs_checkpoint.h5', monitor='accuracy',
                         save_best_only='True', save_weights_only='True', verbose=1)
    es = EarlyStopping(mode='max', monitor='val_acc', patience=10, verbose=1)
    callbacks = [tb, mc, es]

    batch_size = 5
    steps_per_epoch = np.ceil(float(len(frames_list) - round(0.1 * len(frames_list))) / float(batch_size))
    # steps_per_epoch
    validation_steps = (float((round(0.1 * len(frames_list)))) / float(batch_size))
    # validation_steps

    num_epochs = 5
    batch_size = 5
    # result = model.fit_generator(TrainAugmentGenerator(DATA_PATH, id2code, train_frames_datagen,train_masks_datagen), steps_per_epoch=18,
    #                              validation_data=ValAugmentGenerator(DATA_PATH, id2code, val_frames_datagen, val_masks_datagen),
    #                              validation_steps=validation_steps, epochs=num_epochs, callbacks=callbacks)
    #
    # model.save_weights("camvid_model_5_epochs.h5", overwrite=True)

    model.load_weights(model_file)
    # model.load_weights(str('camvid_model_5_epochs.h5', 'utf-8'))
    test_dataset = ValAugmentGenerator(DATA_PATH, id2code, val_frames_datagen, val_masks_datagen)
    print(test_dataset.__next__()[2].shape)
    n = 5
    #ids = np.random.choice(np.arange(len(test_dataset)), size=n)
    #print(ids)
    import cv2
    import time
    store = test_dataset.__next__()
    store_frame = store[0]
    store_mask = store[2]
    for i in range(0,5):

        image, gt_layers = store_frame[i], store_mask[i]
        image = np.expand_dims(image, axis=0)
        start = time.time()
        pr_layers = model.predict(image)
        end = time.time()
        print('Time for forward pass:', end - start)
        pr_mask = seglayers2mask(image, pr_layers)
        gt_mask = seglayers2mask(image, np.expand_dims(gt_layers, axis=0))
        cv2.imwrite("logs/masks_test/gt/gt_mask_" + str(i) + ".jpg", gt_mask)
        cv2.imwrite("logs/masks_test/pr/pr_mask_" + str(i) + ".jpg", pr_mask)
        visualize(
            image=denormalize(image.squeeze()),
            gt_mask=gt_mask,
            pr_mask=pr_mask,
        )

    # scores = model.evaluate_generator(ValAugmentGenerator(DATA_PATH, id2code, val_frames_datagen, val_masks_datagen), \
    #                                   steps=validation_steps, callbacks=callbacks)

    # print("Loss: {:.5}".format(scores[0]))
    # for metric, value in zip(metrics_eval, scores[1:]):
    #     print("mean {}: {:.5}".format(metric.__name__, value))






