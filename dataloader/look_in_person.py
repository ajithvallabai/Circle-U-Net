import os
import tensorflow as tf
import numpy as np

def _read_to_tensor(fname, output_height=256, output_width=256, normalize_data=False):
    '''Function to read images from given image file path, and provide resized images as tensors
        Inputs:
            fname - image file path
            output_height - required output image height
            output_width - required output image width
            normalize_data - if True, normalize data to be centered around 0 (mean 0, range 0 to 1)
        Output: Processed image tensors
    '''

    # Read the image as a tensor
    img_strings = tf.io.read_file(fname)
    imgs_decoded = tf.image.decode_jpeg(img_strings, channels=3)
    print(imgs_decoded.shape)
    # Resize the image
    output = tf.image.resize(imgs_decoded, [output_height, output_width])
    print(output.shape)
    # Normalize if required
    if normalize_data:
        output = (output - 128) / 128
    return output


def read_images(img_dir):
    '''Function to get all image directories, read images and masks in separate tensors
        Inputs:
            img_dir - file directory
        Outputs
            frame_tensors, masks_tensors, frame files list, mask files list
    '''

    # Get the file names list from provided directory
    # file_list = [f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f))]

    file_list_frames = []

    for f in os.listdir(img_dir + "train_frames/train/"):
        file_list_frames.append(f)

    file_list_masks = []

    for f in os.listdir(img_dir + "train_masks/train/"):
        file_list_masks.append(f)

    # Separate frame and mask files lists, exclude unnecessary files
    frames_list = file_list_frames
    masks_list = file_list_masks

    print('{} frame files found in the provided directory.'.format(len(frames_list)))
    print('{} mask files found in the provided directory.'.format(len(masks_list)))

    # Create file paths from file names
    frames_paths = [os.path.join(img_dir + 'train_frames/train/', fname) for fname in frames_list]
    masks_paths = [os.path.join(img_dir + 'train_masks/train/', fname) for fname in masks_list]
    frames_paths.sort()
    masks_paths.sort()

    print(frames_paths)
    print(masks_paths)

    # Create dataset of tensors
    frame_data = tf.data.Dataset.from_tensor_slices(frames_paths)
    masks_data = tf.data.Dataset.from_tensor_slices(masks_paths)

    # Read images into the tensor dataset
    frame_tensors = frame_data.map(_read_to_tensor)
    masks_tensors = masks_data.map(_read_to_tensor)

    print('Completed importing {} frame images from the provided directory.'.format(len(frames_list)))
    print('Completed importing {} mask images from the provided directory.'.format(len(masks_list)))

    return frame_tensors, masks_tensors, frames_list, masks_list


def generate_image_folder_structure(frames, masks, frames_list, masks_list):
    '''Function to save images in the appropriate folder directories
        Inputs:
            frames - frame tensor dataset
            masks - mask tensor dataset
            frames_list - frame file paths
            masks_list - mask file paths
    '''
    # Create iterators for frames and masks
    frame_batches = tf.compat.v1.data.make_one_shot_iterator(
        frames)  # outside of TF Eager, we would use make_one_shot_iterator
    mask_batches = tf.compat.v1.data.make_one_shot_iterator(masks)
    print(frames_list)

    # Iterate over the train images while saving the frames and masks in appropriate folders
    dir_name = 'train'
    for file in zip(frames_list[:-round(0.2 * len(frames_list))], masks_list[:-round(0.2 * len(masks_list))]):
        # Convert tensors to numpy arrays
        frame = frame_batches.next().numpy().astype(np.uint8)
        mask = mask_batches.next().numpy().astype(np.uint8)

        # Convert numpy arrays to images
        frame = Image.fromarray(frame)
        mask = Image.fromarray(mask)

        # Save frames and masks to correct directories
        frame.save(DATA_PATH + '{}_frames/{}'.format(dir_name, dir_name) + '/' + file[0])
        mask.save(DATA_PATH + '{}_masks/{}'.format(dir_name, dir_name) + '/' + file[1])

    # Iterate over the val images while saving the frames and masks in appropriate folders
    dir_name = 'val'
    for file in zip(frames_list[-round(0.2 * len(frames_list)):], masks_list[-round(0.2 * len(masks_list)):]):
        # Convert tensors to numpy arrays
        frame = frame_batches.next().numpy().astype(np.uint8)
        mask = mask_batches.next().numpy().astype(np.uint8)

        # Convert numpy arrays to images
        frame = Image.fromarray(frame)
        mask = Image.fromarray(mask)

        # Save frames and masks to correct directories
        frame.save(DATA_PATH + '{}_frames/{}'.format(dir_name, dir_name) + '/' + file[0])
        mask.save(DATA_PATH + '{}_masks/{}'.format(dir_name, dir_name) + '/' + file[1])

    print("Saved {} frames to directory {}".format(len(frames_list), DATA_PATH))
    print("Saved {} masks to directory {}".format(len(masks_list), DATA_PATH))

def parse_code(l):
    '''Function to parse lines in a text file, returns separated elements (label codes and names in this case)
    '''

    print(l)
    if len(l.strip().split("\t"))==2:
        color_code, object_name = l.strip().split("\t")
        color_code = color_code.split(" ")
        r,g,b = [int(x) for x in color_code]
    else:
        color_code,_, object_name = l.strip().split("\t")
        color_code = color_code.split(" ")
        r, g, b = [int(x) for x in color_code]
    #values = a.split(" ")
    #print(values)
    #values = [int(ele) for ele in values]

    return (r,g,b), object_name

def rgb_to_onehot(rgb_image, colormap):
    '''Function to one hot encode RGB mask labels
        Inputs:
            rgb_image - image matrix (eg. 256 x 256 x 3 dimension numpy ndarray)
            colormap - dictionary of color to label id
        Output: One hot encoded image of dimensions (height x width x num_classes) where num_classes = len(colormap)
    '''
    num_classes = len(colormap)
    shape = rgb_image.shape[:2]+(num_classes,)
    encoded_image = np.zeros( shape, dtype=np.int8 )
    for i, cls in enumerate(colormap):
        encoded_image[:,:,i] = np.all(rgb_image.reshape( (-1,3) ) == colormap[i], axis=1).reshape(shape[:2])
    return encoded_image


def onehot_to_rgb(onehot, colormap):
    '''Function to decode encoded mask labels
        Inputs:
            onehot - one hot encoded image matrix (height x width x num_classes)
            colormap - dictionary of color to label id
        Output: Decoded RGB image (height x width x 3)
    '''
    single_layer = np.argmax(onehot, axis=-1)
    output = np.zeros( onehot.shape[:2]+(3,) )
    for k in colormap.keys():
        output[single_layer==k] = colormap[k]
    return np.uint8(output)

import albumentations as A
from tensorflow.keras.utils import Sequence

def round_clip_0_1(x, **kwargs):
    return x.round().clip(0, 1)

# define heavy augmentations
def get_training_augmentation():
    train_transform = [

        A.HorizontalFlip(p=0.5),

        A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),

        A.PadIfNeeded(min_height=320, min_width=320, always_apply=True, border_mode=0),
        A.RandomCrop(height=320, width=320, always_apply=True),

        A.IAAAdditiveGaussianNoise(p=0.2),
        A.IAAPerspective(p=0.5),

        A.OneOf(
            [
                A.CLAHE(p=1),
                A.RandomBrightness(p=1),
                A.RandomGamma(p=1),
            ],
            p=0.9,
        ),

        A.OneOf(
            [
                A.IAASharpen(p=1),
                A.Blur(blur_limit=3, p=1),
                A.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),

        A.OneOf(
            [
                A.RandomContrast(p=1),
                A.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
        A.Lambda(mask=round_clip_0_1)
    ]
    return A.Compose(train_transform)

# data augumentation
# https://www.kaggle.com/khlevnov/imagedatagenerator-and-albumentations-without-pain
class AugmentDataGenerator(Sequence):
    def __init__(self, datagen, augment=None):
        self.datagen = datagen
        if augment is None:
            self.augment = A.Compose([])
        else:
            self.augment = augment

    def __len__(self):
        return len(self.datagen)

    def __getitem__(self, x):
        images, *rest = self.datagen[x]
        augmented = []
        for image in images:
            image = self.augment(image=image)['image']
            augmented.append(image)
        return (np.array(augmented), *rest)


def TrainAugmentGenerator(DATA_PATH, id2code, train_frames_datagen, train_masks_datagen,seed=1, batch_size=5):
    '''Train Image data generator
        Inputs:
            seed - seed provided to the flow_from_directory function to ensure aligned data flow
            batch_size - number of images to import at a time
        Output: Decoded RGB image (height x width x 3)
    '''
    train_image_generator = train_frames_datagen.flow_from_directory(
        DATA_PATH + 'train_frames/',
        batch_size=batch_size, seed=seed)

    train_mask_generator = train_masks_datagen.flow_from_directory(
        DATA_PATH + 'train_masks/',
        batch_size=batch_size, seed=seed)

    while True:
        X1i = train_image_generator.next()
        X2i = train_mask_generator.next()

        # One hot encoding RGB images
        mask_encoded = [rgb_to_onehot(X2i[0][x, :, :, :], id2code) for x in range(X2i[0].shape[0])]

        yield X1i[0], np.asarray(mask_encoded)


def ValAugmentGenerator(DATA_PATH, id2code, val_frames_datagen, val_masks_datagen, seed=1, batch_size=5):
    '''Validation Image data generator
        Inputs:
            seed - seed provided to the flow_from_directory function to ensure aligned data flow
            batch_size - number of images to import at a time
        Output: Decoded RGB image (height x width x 3)
    '''
    val_image_generator = val_frames_datagen.flow_from_directory(
        DATA_PATH + 'val_frames/',
        batch_size=batch_size, seed=seed)

    val_mask_generator = val_masks_datagen.flow_from_directory(
        DATA_PATH + 'val_masks/',
        batch_size=batch_size, seed=seed)

    while True:
        X1i = val_image_generator.next()
        X2i = val_mask_generator.next()

        # One hot encoding RGB images
        mask_encoded = [rgb_to_onehot(X2i[0][x, :, :, :], id2code) for x in range(X2i[0].shape[0])]

        #yield X1i[0], np.asarray(mask_encoded)
        yield X1i[0], np.asarray(mask_encoded) , X2i[0]
