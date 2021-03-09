import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import argparse
from loss.loss import tversky_loss, dice_coef, dice_coef_loss
import tensorflow as tf
import albumentations as A
from tensorflow.keras.utils import Sequence
import logging
import time
import os
#from Utils.drone_metrics import MeanIoU, IoU, single_class_accuracy, CIoU
from Utils.drone_metrics import *
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def class_tversky(y_true, y_pred):
    smooth = 1

    y_true = K.permute_dimensions(y_true, (3,1,2,0))
    y_pred = K.permute_dimensions(y_pred, (3,1,2,0))

    y_true_pos = K.batch_flatten(y_true)
    y_pred_pos = K.batch_flatten(y_pred)

    y_true_pos = tf.cast(y_true_pos,tf.float64)
    y_pred_pos = tf.cast(y_pred_pos,tf.float64)

    # print("y_true_pos", y_true_pos)
    # print("y_pred_pos",y_pred_pos)
    # print("y_true_pos type", type(y_true_pos))
    # print("y_pred_pos type",type(y_pred_pos))

    # print("y_true_pos dtype", y_true_pos.dtype)
    # print("y_pred_pos dtype",y_pred_pos.dtype)

    true_pos = K.sum(y_true_pos * y_pred_pos, 1)
    false_neg = K.sum(y_true_pos * (1-y_pred_pos), 1)
    false_pos = K.sum((1-y_true_pos)*y_pred_pos, 1)
    alpha = 0.7
    return (true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)

def focal_tversky_loss(y_true,y_pred):
    pt_1 = class_tversky(y_true, y_pred)
    gamma = 0.75
    return K.sum(K.pow((1-pt_1), gamma))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset",required=True,
                    help="dataset")
    ap.add_argument("-idir", "--img_directory", required=True,
                    help="image directory")
    ap.add_argument("-m", "--model", required=True,
                    help="model")
    ap.add_argument("-ht", "--output_height", required=False, default=256,
                    help="output height")
    ap.add_argument("-w", "--output_width", required=False, default=256,
                    help="output width")
    ap.add_argument("-c", "--classes", required=False, default=24,
                    help="number of classes")
    ap.add_argument("-bs", "--batch_size", required=False, default=5,
                    help="batchsize")
    ap.add_argument("-l", "--loss", required=True, default="tversky",
                    help="loss function")
    ap.add_argument("-n", "--num_epochs", required=True, default=60,
                    help="total epochs")
    args = vars(ap.parse_args())
    #print(args)



    # get as arguments
    dataset = args['dataset']
    img_dir = args['img_directory']
    DATA_PATH = args['img_directory']
    model_type = args['model']
    loss_function_type = args['loss']
    total_num_epochs = int(args['num_epochs'])

    if loss_function_type == "tversky" or loss_function_type == "CCE":
        pass
    else:
        print("Loss function is doesnt support")

    # Required image dimensions
    output_height = args['output_height']
    output_width = args['output_width']
    #print(dataset,img_dir,model_type)
    # log file declarations
    logname = "logs/complete/" + str(time.time()) + "_logfile"
    logging.basicConfig(filename=logname,
                        filemode='a',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.DEBUG)

    logging.info("Dataset :" + dataset)
    logging.info("Model name :" + model_type )

    # loading dataloader in accordance with dataset
    if dataset == "look_in_person" :
        from dataloader.look_in_person import read_images, parse_code
        from dataloader.look_in_person import TrainAugmentGenerator, ValAugmentGenerator
    elif dataset == "camvid_small":
        from dataloader.camvid_small import read_images, parse_code
        from dataloader.camvid_small import TrainAugmentGenerator, ValAugmentGenerator
    elif dataset == "camvid_full":
        from dataloader.camvid_full import read_images, parse_code
        from dataloader.camvid_full import TrainAugmentGenerator, ValAugmentGenerator


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


    # Seed defined for aligning images and their masks
    seed = 1

    batch_size = int(args['batch_size'])
    classes = args['classes']

    if model_type == "tiny_unet":
        from models.small_unet import UNet
        model = UNet(n_filters = classes)
    elif model_type == "squeeze_unet_tf":
        from models.squeeze_unet_tf import UNet
        model = UNet(batch_size,classes)
    elif model_type == "squeeze_unet_keras":
        from models.squeeze_unet_keras import UNet
        model = UNet(batch_size=batch_size,classes=classes)
    elif model_type == "sq_att_unet":
        from models.sq_att_unet import UNet
        model = UNet(batch_size=batch_size,classes=classes)
    elif model_type == "sq_r_att_unet":
        from models.recurrent_unet import UNet
        model = UNet(batch_size=batch_size,classes=classes)
    elif model_type == ("hr_net"):
        from models.hr_net import HRNet
        model = HRNet(nClasses=classes)
    elif model_type == ("att_unet"):
        from models.att_unet import UNet
        model = UNet(n_filters=32)
    elif model_type == ("res_unet"):
        from models.res_unet import unet_resnet_101
        model = unet_resnet_101(height=256, width=256, channel=3, classes=classes)
    elif model_type == ("circlenet"):
        from models.circle_unet import Circle_unet_resnet_101
        model = Circle_unet_resnet_101(height=256, width=256, channel=3, classes=classes)
    elif model_type == ("circle_att_101"):
        from models.circle_attention import circle_att_101
        model = circle_att_101(height=256, width=256, channel=3, classes=classes)
    elif model_type == ("new_squeezenet"):
        from models.nsqueeze_unet import SqueezeUNet
        model = SqueezeUNet(num_classes=24)
        
    elif model_type == ("psp_net"):
        from models.psp_net2 import _pspnet
        from models.PSP_Net.models.basic_models import vanilla_encoder
        model = _pspnet(101, vanilla_encoder, input_height=384, input_width=384)

    ## DEFINE METRICS AND WEIGHTS LOSS FUNCTION ##
    num_classes = 24
    miou_metric = MeanIoU(num_classes)
    unlabeled_iou_metric = IoU0(num_classes, 0,"unlabeled")

    paved_iou_metric = IoU1(num_classes, 1,"paved-area")

    dirt_iou_metric = IoU2(num_classes, 2,"dirt")
    grass_iou_metric = IoU3(num_classes, 3,"grass")
    gravel_iou_metric = IoU4(num_classes, 4,"gravel")
    water_iou_metric = IoU5(num_classes, 5,"water")
    rocks_iou_metric = IoU6(num_classes, 6,"rocks")
    pool_iou_metric = IoU7(num_classes, 7,"pool")
    vegetation_iou_metric = IoU8(num_classes, 8,"vegetation")
    roof_iou_metric = IoU9(num_classes, 9,"roof")
    wall_iou_metric = IoU10(num_classes, 10,"wall")
    window_iou_metric = IoU11(num_classes, 11,"window")
    door_iou_metric = IoU12(num_classes, 12,"door")
    fence_iou_metric = IoU13(num_classes, 13,"fence")
    fence_pole_iou_metric = IoU14(num_classes, 14,"fence-pole")
    person_iou_metric = IoU15(num_classes, 15,"person")
    dog_iou_metric = IoU16(num_classes, 16,"dog")
    car_iou_metric = IoU17(num_classes, 17,"car")
    bicycle_iou_metric = IoU18(num_classes, 18,"bicycle")
    tree_iou_metric = IoU19(num_classes, 19,"tree")
    bald_tree_iou_metric = IoU20(num_classes, 20,"bald-tree")
    ar_marker_iou_metric = IoU21(num_classes, 21,"ar-marker")
    obstacle_iou_metric = IoU22(num_classes, 22,"obstacle")
    conflicting_iou_metric = IoU23(num_classes, 23,"conflicting")

    # need to change below names
    unlabeled_acc_metric = single_class_accuracy(0)
    unlabeled_acc_metric.__name__ = "unlabeled_acc"

    paved_acc_metric = single_class_accuracy(1)
    paved_acc_metric.__name__ = "paved_acc"

    dirt_acc_metric = single_class_accuracy(2)
    dirt_acc_metric.__name__ = "dirt_acc"

    grass_acc_metric = single_class_accuracy(3)
    grass_acc_metric.__name__ = "grass_acc"

    gravel_acc_metric = single_class_accuracy(4)
    gravel_acc_metric.__name__ = "gravel_acc"

    water_acc_metric = single_class_accuracy(5)
    water_acc_metric.__name__ = "water_acc"

    rocks_acc_metric = single_class_accuracy(6)
    rocks_acc_metric.__name__ = "rocks_acc"

    pool_acc_metric = single_class_accuracy(7)
    pool_acc_metric.__name__ = "pool_acc"

    vegetation_acc_metric = single_class_accuracy(8)
    vegetation_acc_metric.__name__ = "vegetation_acc"

    roof_acc_metric = single_class_accuracy(9)
    roof_acc_metric.__name__ = "roof_acc"

    wall_acc_metric = single_class_accuracy(10)
    wall_acc_metric.__name__ = "wall_acc"

    window_acc_metric = single_class_accuracy(11)
    window_acc_metric.__name__ = "window_acc"

    door_acc_metric = single_class_accuracy(12)
    door_acc_metric.__name__ = "door_acc"

    fence_acc_metric = single_class_accuracy(13)
    fence_acc_metric.__name__ = "fence_acc"

    fence_pole_acc_metric = single_class_accuracy(14)
    fence_pole_acc_metric.__name__ = "fence_pole_acc"

    person_acc_metric = single_class_accuracy(15)
    person_acc_metric.__name__ = "person_acc"

    dog_acc_metric = single_class_accuracy(16)
    dog_acc_metric.__name__ = "dog_acc"

    car_acc_metric = single_class_accuracy(17)
    car_acc_metric.__name__ = "car_acc"

    bicycle_acc_metric = single_class_accuracy(18)
    bicycle_acc_metric.__name__ = "bicycle_acc"

    tree_acc_metric = single_class_accuracy(19)
    tree_acc_metric.__name__ = "tree_acc"

    bald_tree_acc_metric = single_class_accuracy(20)
    bald_tree_acc_metric.__name__ = "bald_tree_acc"

    ar_marker_acc_metric = single_class_accuracy(21)
    ar_marker_acc_metric.__name__ = "ar_marker_acc"

    obstacle_acc_metric = single_class_accuracy(22)
    obstacle_acc_metric.__name__ = "obstacle_acc"

    conflicting_acc_metric = single_class_accuracy(23)
    conflicting_acc_metric.__name__ = "conflicting_acc"


    # Compiling model
    # model.compile(optimizer='adam', loss=,  \
    #                                 metrics = ['accuracy',miou_metric.mean_iou,void_iou_metric.iou,sky_iou_metric.iou] )

    iou_acc_metrics = ['accuracy', \
     miou_metric.mean_iou, unlabeled_iou_metric.iou, paved_iou_metric.iou, \
        dirt_iou_metric.iou , grass_iou_metric.iou, gravel_iou_metric.iou, \

        water_iou_metric.iou , rocks_iou_metric.iou, pool_iou_metric.iou, \
        vegetation_iou_metric.iou , roof_iou_metric.iou, wall_iou_metric.iou, \
        window_iou_metric.iou, door_iou_metric.iou, fence_iou_metric.iou ,\
        fence_pole_iou_metric.iou, person_iou_metric.iou,  dog_iou_metric.iou, \
        car_iou_metric.iou, bicycle_iou_metric.iou , tree_iou_metric.iou, bald_tree_iou_metric.iou, \
        ar_marker_iou_metric.iou, obstacle_iou_metric.iou, conflicting_iou_metric.iou, \

     unlabeled_acc_metric, paved_acc_metric, dirt_acc_metric , grass_acc_metric, \
    gravel_acc_metric, water_acc_metric, rocks_acc_metric, pool_acc_metric, \
    vegetation_acc_metric, roof_acc_metric, wall_acc_metric, window_acc_metric, \
    door_acc_metric, fence_acc_metric, fence_pole_acc_metric, person_acc_metric, \
    dog_acc_metric, car_acc_metric, bicycle_acc_metric, tree_acc_metric, \
    bald_tree_acc_metric, ar_marker_acc_metric, obstacle_acc_metric, conflicting_acc_metric,\
     ]
    # model.compile(optimizer='adam', loss="categorical_crossentropy", \
    #               metrics=iou_acc_metrics)

    if loss_function_type=="tversky":
        #focal_tversky_loss
        print("Focal tversky loss")
        model.compile(optimizer='adam', loss=focal_tversky_loss, \
                    metrics=iou_acc_metrics)
    elif loss_function_type== "CCE":
        #CCE
        model.compile(optimizer='adam', loss="categorical_crossentropy", \
                   metrics=iou_acc_metrics)
    num_epochs = total_num_epochs

    #model.summary()
    tb = TensorBoard(log_dir='logs', write_graph=True)
#     mc = ModelCheckpoint(mode='max', filepath='camvid_model_epochs_checkpoint.h5', monitor='accuracy',
#                          save_best_only='True', save_weights_only='True', verbose=1)
    #keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', #period=1)
    filepath = "weights/2021_02_21_saved-" + model_type +"model_" + str(num_epochs) + "_plus-{epoch:02d}-{mean_iou:.2f}.hdf5"
    mc = ModelCheckpoint(filepath, monitor='mean_iou', verbose=1, save_best_only=True, mode='max')

    es = EarlyStopping(mode='max', monitor='mean_iou', patience=10, verbose=1)

    # callbacks = [tb, mc, es]
    callbacks = [tb, es]



    steps_per_epoch = np.ceil(float(len(frames_list) - round(0.1 * len(frames_list))) / float(batch_size))
    #steps_per_epoch
    validation_steps = (float((round(0.1 * len(frames_list)))) / float(batch_size))
    # validation_steps

    


    # Data augumentation
    # Normalizing only frame images, since masks contain label info
    # data_gen_args = dict(rescale=1. / 255,preprocessing_function=transform)
    # mask_gen_args = dict(preprocessing_function=transform)

#     data_gen_args = dict(rescale=1. / 255)
#     mask_gen_args = dict()

    data_gen_args = dict(rescale=1. / 255,height_shift_range=0.15, rotation_range=10)
    mask_gen_args = dict( height_shift_range=0.15, rotation_range=10)
    logging.info("loss function used -", loss_function_type) 
    logging.info("Data augumentation used for frame - ", data_gen_args)
    logging.info("Data augumentation used for mask - ", mask_gen_args)
    
    # data_gen_args = dict(rescale=1. / 255, featurewise_center=True, height_shift_range=0.15, \
    #                      brightness_range=(0.7, 0.9), zoom_range=[0.5, 1.5], \
    #                      rotation_range=10, )
    # mask_gen_args = dict(featurewise_center=True, height_shift_range=0.15, \
    #                      brightness_range=(0.7, 0.9), zoom_range=[0.5, 1.5], \
    #                      rotation_range=10, )

    val_data_gen_args =  dict(rescale=1. / 255)
    val_mask_gen_args = dict()

    train_frames_datagen = ImageDataGenerator(**data_gen_args)
    train_masks_datagen = ImageDataGenerator(**mask_gen_args)
    val_frames_datagen = ImageDataGenerator(**val_data_gen_args)
    val_masks_datagen = ImageDataGenerator(**val_mask_gen_args)

    train_datagen = TrainAugmentGenerator(DATA_PATH, id2code, train_frames_datagen,train_masks_datagen,batch_size=batch_size)
    val_datagen = ValAugmentGenerator(DATA_PATH, id2code, val_frames_datagen, val_masks_datagen,batch_size=batch_size)

    # using Ablumenations library for Data augumentation
    # train_datagen = AugmentDataGenerator(train_datagen, get_training_augmentation())
    #
    # val_datagen = AugmentDataGenerator(val_datagen,None)
    # val_frames_datagen = AugmentDataGenerator(val_frames_datagen )
    # val_masks_datagen = AugmentDataGenerator(val_masks_datagen)

    
    print("batch size", batch_size)
    #model.load_weights("weights/2021_02_21_saved-Resnet101model-30-0.19.hdf5")
    

        
    result = model.fit_generator(train_datagen, steps_per_epoch=steps_per_epoch,
                                 validation_data=val_datagen,
                                 validation_steps=validation_steps, epochs=num_epochs, \
                                 callbacks=callbacks, \
                                 verbose=1,
                                 )
    history_result = result.history
    logging.info("Training logs")
    print(history_result)
    logging.info(history_result)
    print(history_result.keys())
    for each in history_result.keys():
        logging.info(each + ":" + str(history_result[each]))

    model.save_weights("weights/Final_"+ model_type + "_full_model_" + str(num_epochs) +"_epochs.h5", overwrite=True)
    #2021_02_21_saved-Resnet101model-30-0.19.hdf5

    #model.load_weights("camvid_model_2_epochs.h5")
    #model.load_weights(str('weights/camvid_model_1_epochs.h5', 'utf-8'))

    scores = model.evaluate_generator(ValAugmentGenerator(DATA_PATH, id2code, val_frames_datagen, val_masks_datagen),\
                                      steps=validation_steps,callbacks=callbacks)
    print(scores)
    logging.info("Testing logs")
    for each in scores:
        logging.info("scores_"+ str(each) )


    logging.info("Testing final results")

    print("Loss: {:.5}".format(scores[0]))
    # logging.info("Loss: {:.5}".format(scores[0]))
    # for metric, value in zip(metrics_eval, scores[1:]):
    #     print("mean {}: {:.5}".format(metric.__name__, value))
    #     logging.info("mean {}: {:.5}".format(metric.__name__, value))




