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
from Utils.camvid_metrics import *
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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
                    help="output width")
    ap.add_argument("-bs", "--batch_size", required=False, default=5,
                    help="output width")
    args = vars(ap.parse_args())
    #print(args)



    # get as arguments
    dataset = args['dataset']
    img_dir = args['img_directory']
    DATA_PATH = args['img_directory']
    model_type = args['model']

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
    elif dataset == "camvid_small" or dataset == "iug_drone":
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
    classes = int(args['classes'])

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
    elif model_type == ("psp_net"):
        from models.psp_net2 import _pspnet
        from models.PSP_Net.models.basic_models import vanilla_encoder
        model = _pspnet(101, vanilla_encoder, input_height=384, input_width=384)

    ## DEFINE METRICS AND WEIGHTS LOSS FUNCTION ##
    num_classes = 32
    miou_metric = MeanIoU(num_classes)
    Animal_iou_metric = IoU0(num_classes, 0,"Animal")
    Archway_iou_metric = IoU1(num_classes, 1,"Archway")
    Bicyclist_iou_metric = IoU2(num_classes, 2,"Bicyclist")
    Bridge_iou_metric = IoU3(num_classes, 3,"Bridge")
    Building_iou_metric = IoU4(num_classes, 4,"Building")
    Car_iou_metric = IoU5(num_classes, 5,"Car")
    CartLuggagePram_iou_metric = IoU6(num_classes, 6,"CartLuggagePram")
    Child_iou_metric = IoU7(num_classes, 7,"Child")
    Column_Pole_iou_metric = IoU8(num_classes, 8,"Column_Pole")
    Fence_iou_metric = IoU9(num_classes, 9,"Fence")
    LaneMkgsDriv_iou_metric = IoU10(num_classes, 10,"LaneMkgsDriv")
    LaneMkgsNonDriv_iou_metric = IoU11(num_classes, 11,"LaneMkgsNonDriv")
    Misc_Text_iou_metric = IoU12(num_classes, 12,"Misc_Text")
    MotorcycleScooter_iou_metric = IoU13(num_classes, 13,"MotorcycleScooter")
    OtherMoving_iou_metric = IoU14(num_classes, 14,"OtherMoving")
    ParkingBlock_iou_metric = IoU15(num_classes, 15,"ParkingBlock")
    Pedestrian_iou_metric = IoU16(num_classes, 16,"Pedestrian")
    Road_iou_metric = IoU17(num_classes, 17,"Road")
    RoadShoulder_iou_metric = IoU18(num_classes, 18,"RoadShoulder")
    Sidewalk_iou_metric = IoU19(num_classes, 19,"Sidewalk")
    SignSymbol_iou_metric = IoU20(num_classes, 20,"SignSymbol")
    Sky_iou_metric = IoU21(num_classes, 21,"Sky")
    SUVPickupTruck_iou_metric = IoU22(num_classes, 22,"SUVPickupTruck")
    TrafficCone_iou_metric = IoU23(num_classes, 23,"TrafficCone")
    TrafficLight_iou_metric = IoU24(num_classes, 24, "TrafficLight")
    Train_iou_metric = IoU25(num_classes, 25, "Train")
    Tree_pole_iou_metric = IoU26(num_classes, 26, "Tree")
    Truck_Bus_iou_metric = IoU27(num_classes, 27, "Truck_Bus")
    Tunnel_iou_metric = IoU28(num_classes, 28, "Tunnel")
    VegetationMisc_iou_metric = IoU29(num_classes, 29, "VegetationMisc")
    Void_iou_metric = IoU30(num_classes, 30, "Void")
    Wall_iou_metric = IoU31(num_classes, 31, "Wall")



    # need to change below names
    Animal_acc_metric = single_class_accuracy(0)
    Animal_acc_metric.__name__ = "Animal"

    Archway_acc_metric = single_class_accuracy(1)
    Archway_acc_metric.__name__ = "Archway"

    Bicyclist_acc_metric = single_class_accuracy(2)
    Bicyclist_acc_metric.__name__ = "Bicyclist"

    Bridge_acc_metric = single_class_accuracy(3)
    Bridge_acc_metric.__name__ = "Bridge"

    Building_acc_metric = single_class_accuracy(4)
    Building_acc_metric.__name__ = "Building"

    Car_acc_metric = single_class_accuracy(5)
    Car_acc_metric.__name__ = "Car"

    CartLuggagePram_acc_metric = single_class_accuracy(6)
    CartLuggagePram_acc_metric.__name__ = "CartLuggagePram"

    Child_acc_metric = single_class_accuracy(7)
    Child_acc_metric.__name__ = "Child"

    Column_Pole_acc_metric = single_class_accuracy(8)
    Column_Pole_acc_metric.__name__ = "Column_Pole"

    Fence_acc_metric = single_class_accuracy(9)
    Fence_acc_metric.__name__ = "Fence"

    LaneMkgsDriv_acc_metric = single_class_accuracy(10)
    LaneMkgsDriv_acc_metric.__name__ = "LaneMkgsDriv"

    LaneMkgsNonDriv_acc_metric = single_class_accuracy(11)
    LaneMkgsNonDriv_acc_metric.__name__ = "LaneMkgsNonDriv"

    Misc_Text_acc_metric = single_class_accuracy(12)
    Misc_Text_acc_metric.__name__ = "Misc_Text"

    MotorcycleScooter_acc_metric = single_class_accuracy(13)
    MotorcycleScooter_acc_metric.__name__ = "MotorcycleScooter"

    OtherMoving_acc_metric = single_class_accuracy(14)
    OtherMoving_acc_metric.__name__ = "OtherMoving"

    ParkingBlock_acc_metric = single_class_accuracy(15)
    ParkingBlock_acc_metric.__name__ = "ParkingBlock"

    Pedestrian_acc_metric = single_class_accuracy(16)
    Pedestrian_acc_metric.__name__ = "Pedestrian"

    Road_acc_metric = single_class_accuracy(17)
    Road_acc_metric.__name__ = "Road"

    RoadShoulder_acc_metric = single_class_accuracy(18)
    RoadShoulder_acc_metric.__name__ = "RoadShoulder"

    Sidewalk_acc_metric = single_class_accuracy(19)
    Sidewalk_acc_metric.__name__ = "Sidewalk"

    SignSymbol_acc_metric = single_class_accuracy(20)
    SignSymbol_acc_metric.__name__ = "SignSymbol"

    Sky_acc_metric = single_class_accuracy(21)
    Sky_acc_metric.__name__ = "Sky"

    SUVPickupTruck_acc_metric = single_class_accuracy(22)
    SUVPickupTruck_acc_metric.__name__ = "SUVPickupTruck"

    TrafficCone_acc_metric = single_class_accuracy(23)
    TrafficCone_acc_metric.__name__ = "TrafficCone"

    TrafficLight_acc_metric = single_class_accuracy(24)
    TrafficLight_acc_metric.__name__ = "TrafficLight"

    Train_acc_metric = single_class_accuracy(25)
    Train_acc_metric.__name__ = "Train"

    Tree_acc_metric = single_class_accuracy(26)
    Tree_acc_metric.__name__ = "Tree"

    Truck_Bus_acc_metric = single_class_accuracy(27)
    Truck_Bus_acc_metric.__name__ = "Truck_Bus"

    Tunnel_acc_metric = single_class_accuracy(28)
    Tunnel_acc_metric.__name__ = "Tunnel"

    VegetationMisc_acc_metric = single_class_accuracy(29)
    VegetationMisc_acc_metric.__name__ = "VegetationMisc"

    Void_acc_metric = single_class_accuracy(30)
    Void_acc_metric.__name__ = "Void"

    Wall_acc_metric = single_class_accuracy(31)
    Wall_acc_metric.__name__ = "Wall"

    # Compiling model
    # model.compile(optimizer='adam', loss=,  \
    #                                 metrics = ['accuracy',miou_metric.mean_iou,void_iou_metric.iou,sky_iou_metric.iou] )

    iou_acc_metrics = ['accuracy', \
    miou_metric.mean_iou, Animal_iou_metric.iou, Archway_iou_metric.iou, \
    Bicyclist_iou_metric.iou , Bridge_iou_metric.iou, Building_iou_metric.iou, \
    Car_iou_metric.iou , CartLuggagePram_iou_metric.iou, Child_iou_metric.iou, \
    Column_Pole_iou_metric.iou , Fence_iou_metric.iou, LaneMkgsDriv_iou_metric.iou, \
    LaneMkgsDriv_iou_metric.iou, Misc_Text_iou_metric.iou, MotorcycleScooter_iou_metric.iou ,\
    OtherMoving_iou_metric.iou, ParkingBlock_iou_metric.iou,  Pedestrian_iou_metric.iou, \
    Road_iou_metric.iou, RoadShoulder_iou_metric.iou , Sidewalk_iou_metric.iou, SignSymbol_iou_metric.iou, \
    Sky_iou_metric.iou, SUVPickupTruck_iou_metric.iou, TrafficCone_iou_metric.iou, \
    TrafficLight_iou_metric.iou, Train_iou_metric.iou, Tree_pole_iou_metric.iou, \
    Truck_Bus_iou_metric.iou, Tunnel_iou_metric.iou, VegetationMisc_iou_metric.iou, Void_iou_metric.iou, \
    Wall_iou_metric.iou, \

    Animal_acc_metric, Animal_acc_metric, Bicyclist_acc_metric , Bridge_acc_metric, \
    Building_acc_metric, Car_acc_metric, CartLuggagePram_acc_metric, Child_acc_metric, \
    Column_Pole_acc_metric, Fence_acc_metric, LaneMkgsDriv_acc_metric, LaneMkgsNonDriv_acc_metric, \
    Misc_Text_acc_metric, MotorcycleScooter_acc_metric, OtherMoving_acc_metric, ParkingBlock_acc_metric, \
    Pedestrian_acc_metric, Road_acc_metric, RoadShoulder_acc_metric, Sidewalk_acc_metric, \
    Sidewalk_acc_metric, SignSymbol_acc_metric, Sky_acc_metric, SUVPickupTruck_acc_metric,\
    TrafficCone_acc_metric, TrafficLight_acc_metric, Train_acc_metric, Tree_acc_metric, \
    Truck_Bus_acc_metric, Tunnel_acc_metric, VegetationMisc_acc_metric,Void_acc_metric, \
    Wall_acc_metric
    ]

    model.compile(optimizer='adam', loss="categorical_crossentropy", \
                  metrics=iou_acc_metrics)

    #model.summary()
    tb = TensorBoard(log_dir='logs', write_graph=True)
    mc = ModelCheckpoint(mode='max', filepath='camvid_model_epochs_checkpoint.h5', monitor='accuracy',
                         save_best_only='True', save_weights_only='True', verbose=1)
    es = EarlyStopping(mode='max', monitor='val_acc', patience=10, verbose=1)

    callbacks = [tb, mc, es]



    steps_per_epoch = np.ceil(float(len(frames_list) - round(0.1 * len(frames_list))) / float(batch_size))
    #steps_per_epoch
    validation_steps = (float((round(0.1 * len(frames_list)))) / float(batch_size))
    # validation_steps

    num_epochs = 10


    # Data augumentation
    # Normalizing only frame images, since masks contain label info
    # data_gen_args = dict(rescale=1. / 255,preprocessing_function=transform)
    # mask_gen_args = dict(preprocessing_function=transform)

    data_gen_args = dict(rescale=1. / 255,)
    mask_gen_args = dict()

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

    num_epochs = 1
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

    model.save_weights("camvid_full_model_10_epochs.h5", overwrite=True)

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




