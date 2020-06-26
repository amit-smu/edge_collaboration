from keras import backend as K
from keras.models import load_model
from keras.optimizers import Adam
from scipy.misc import imread
import numpy as np
from matplotlib import pyplot as plt

from models.keras_ssd512 import ssd_512
from keras_loss_function.keras_ssd_loss import SSDLoss
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast
from keras_layers.keras_layer_L2Normalization import L2Normalization
from data_generator.object_detection_2d_data_generator_multi_resolution_evaluation_512 import DataGenerator
from eval_utils.average_precision_evaluator_single_image_multi_resolution import Evaluator

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# # Set a few configuration parameters.
img_height = 512
img_width = 512
n_classes = 20  # other than background
model_mode = 'inference'

#
# ####################################     Load Weights/MODEL here    ##########################################
# # 1: Build the Keras model
K.clear_session()  # Clear previous models from memory.

model = ssd_512(image_size=(img_height, img_width, 3),
                n_classes=20,
                mode='inference',
                l2_regularization=0.0005,
                scales=[0.07, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05],
                # The scales for MS COCO are [0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05]
                aspect_ratios_per_layer=[[1.0, 2.0, 0.5],
                                         [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                         [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                         [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                         [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                         [1.0, 2.0, 0.5],
                                         [1.0, 2.0, 0.5]],
                two_boxes_for_ar1=True,
                steps=[8, 16, 32, 64, 128, 256, 512],
                offsets=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                clip_boxes=False,
                variances=[0.1, 0.1, 0.2, 0.2],
                normalize_coords=True,
                subtract_mean=[123, 117, 104],
                swap_channels=[2, 1, 0],
                confidence_thresh=0.5,
                iou_threshold=0.45,
                top_k=200,
                nms_max_output_size=400)

# 2: Load the trained weights into the model.

# TODO: Set the path of the trained weights.
weights_path = './single_img_models/ssd512_PETS+WT_person_180_epoch-179_loss-2.8713_val_loss-2.7450.h5'
# weights_path = './single_img_models/ssd512_pascal_PETS+WT_all_classes_epoch-118_loss-3.0553_val_loss-2.8864.h5'
# weights_path = './trained_models/VGG_VOC0712_SSD_512x512_iter_120000.h5'

print(weights_path + "\n")

model.load_weights(weights_path, by_name=True)

# 3: Compile the model so that Keras won't complain the next time you load it.

adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)

model.compile(optimizer=adam, loss=ssd_loss.compute_loss)

############################################### VARIABLES #######################
test_dataset = "WILDTRACK"
#test_dataset = "PETS"
# test_dataset = "VOC"
test_img_resolution = 512
###############################################    DATA GENERATORS     ################################################


# TODO: Set the paths to the dataset here.
# Pascal_VOC_dataset_images_dir = '../dataset/PASCAL_VOC/VOC2007/VOCdevkit/VOC2007/JPEGImages/'
# Pascal_VOC_dataset_annotations_dir = '../dataset/PASCAL_VOC/VOC2007/VOCdevkit/VOC2007/Annotations/'
# Pascal_VOC_dataset_image_set_filename = '../dataset/PASCAL_VOC/VOC2007/VOCdevkit/VOC2007/ImageSets/Main/test_subset_500.txt'

# PETS_images_dir = '../dataset/PETS_org/JPEGImages'
# PETS_annotations_dir = '../dataset/PETS_org/Annotations'
# PETS_test_image_set_filename = '../dataset/PETS_org/ImageSets/Main/test_12.txt'

PETS_images_dir = '../dataset/PETS_org/JPEGImages_cropped_400x400'
PETS_annotations_dir = '../dataset/PETS_org/Annotations_cropped_400x400'
PETS_test_image_set_filename = '../dataset/PETS_org/ImageSets/Main/test_crop_400x400.txt'

WT_dataset_images_dir = "../dataset/Wildtrack_dataset/PNGImages_cropped_700x700"
WT_dataset_annotations_dir = "../dataset/Wildtrack_dataset/Annotations_cropped_700x700"
WT_dataset_test_image_set_filename = "../dataset/Wildtrack_dataset/ImageSets/Main/test_crop_700x700_cam_1.txt"
#
# WT_dataset_images_dir = "../dataset/Wildtrack_dataset/PNGImages"
# WT_dataset_annotations_dir = "../dataset/Wildtrack_dataset/Annotations"
# WT_dataset_test_image_set_filename = "../dataset/Wildtrack_dataset/ImageSets/Main/test.txt"

# The XML parser needs to now what object class names to look for and in which order to map them to integers.
classes = ['background',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat',
           'chair', 'cow', 'diningtable', 'dog',
           'horse', 'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor']
# classes = ['background', 'person']


print("Evaluating for dataset : {}\n".format(test_dataset))


dataset = DataGenerator(load_images_into_memory=True, hdf5_dataset_path=None, resolution=test_img_resolution)
if test_dataset == "PETS":
    dataset.parse_xml(images_dirs=[PETS_images_dir],
                      image_set_filenames=[PETS_test_image_set_filename],
                      annotations_dirs=[PETS_annotations_dir],
                      classes=classes,
                      include_classes='all',
                      exclude_truncated=False,
                      exclude_difficult=False,
                      ret=False)
if test_dataset == "WILDTRACK":
    dataset.parse_xml(images_dirs=[WT_dataset_images_dir],
                      image_set_filenames=[WT_dataset_test_image_set_filename],
                      annotations_dirs=[WT_dataset_annotations_dir],
                      classes=classes,
                      # include_classes='all',
                      include_classes=[15],
                      exclude_truncated=False,
                      exclude_difficult=False,
                      ret=False)

print("testing for : {}\n".format(test_dataset))
    ###############################################    EVALUATION     ################################################
avg_precisions = []
for i in range(1):
    evaluator = Evaluator(model=model,
                          n_classes=n_classes,
                          data_generator=dataset,
                          model_mode=model_mode)

    results = evaluator(img_height=img_height,
                        img_width=img_width,
                        batch_size=16,
                        data_generator_mode='resize',
                        round_confidences=False,
                        matching_iou_threshold=0.5,
                        border_pixels='include',
                        sorting_algorithm='quicksort',
                        average_precision_mode='sample',
                        num_recall_points=11,
                        ignore_neutral_boxes=True,
                        return_precisions=True,
                        return_recalls=True,
                        return_average_precisions=True,
                        verbose=True)

    mean_average_precision, average_precisions, precisions, recalls = results

    # print(results)

    ###############################################    VISUALISE RESULTS     ##############################################

    for i in range(15, 16):
        a_prec = round(average_precisions[i], 3)
        print("{:<14}{:<6}{}".format(classes[i], 'AP', a_prec))
    print()
    print("{:<14}{:<6}{}".format('', 'mAP', round(mean_average_precision, 3)))
    avg_precisions.append(a_prec)

print("all_scores: {}".format(avg_precisions))
print("average AP : {}".format(np.max(avg_precisions)))
