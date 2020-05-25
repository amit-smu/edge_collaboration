"""
test script for auxillary input models. Simply load model and draw the predictions on the test dataset
"""

import cv2
from keras import backend as K
from keras.models import load_model
from keras.optimizers import Adam
from scipy.misc import imread
import numpy as np
from matplotlib import pyplot as plt

from keras.preprocessing import image
from models.keras_ssd300_aux import ssd_300_aux
from keras_loss_function.keras_ssd_loss import SSDLoss
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from ssd_encoder_decoder.ssd_output_decoder import decode_detections, decode_detections_fast
from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast
from keras_layers.keras_layer_L2Normalization import L2Normalization
from data_generator.object_detection_2d_data_generator_aux_multi_resolution_evaluation import DataGenerator
from eval_utils.average_precision_evaluator_aux_multi_resolution import Evaluator
import os
from PIL import Image
from xml.etree import ElementTree
import random

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
test_resolution = (300, 300)


def translate_coordinates(coordinates, org_width, org_height):
    xmin1 = int((coordinates[0] / org_width) * test_resolution[0])
    ymin1 = int((coordinates[1] / org_height) * test_resolution[1])
    xmax1 = int((coordinates[2] / org_width) * test_resolution[0])
    ymax1 = int((coordinates[3] / org_height) * test_resolution[1])

    return xmin1, ymin1, xmax1, ymax1


def get_auxillary_channel(xml_file_path, randomize):
    aux_channel = np.full((img_height, img_width, 1), fill_value=0, dtype=np.uint8)

    SHIFT_RANGE = 50  # number of pixels to shift
    WIDTH_RANGE = 10
    HEIGHT_RANGE = 10
    aux_height, aux_width, _ = aux_channel.shape
    # load object coordinates from xml
    root = ElementTree.parse(xml_file_path).getroot()

    # find orginial size of annotation file
    size = root.find("size")
    annot_width = int(size[0].text)
    annot_height = int(size[1].text)

    objects = root.findall("object")
    for obj in objects:
        if obj[0].text == "person":
            # bndbox = obj[4]
            bndbox = obj.find('bndbox')
            xmin = int(float(bndbox[0].text))
            ymin = int(float(bndbox[1].text))
            xmax = int(float(bndbox[2].text))
            ymax = int(float(bndbox[3].text))

            xmin, ymin, xmax, ymax = translate_coordinates([xmin, ymin, xmax, ymax], org_width=annot_width,
                                                           org_height=annot_height)

            if randomize:
                width = xmax - xmin
                height = ymax - ymin

                # random perturbations
                shift_amt = random.randint(-1 * SHIFT_RANGE, SHIFT_RANGE)
                shift_amt_y = random.randint(-1 * SHIFT_RANGE, SHIFT_RANGE)
                width_amt = random.randint(-1 * WIDTH_RANGE, WIDTH_RANGE)
                height_amt = random.randint(-1 * HEIGHT_RANGE, HEIGHT_RANGE)

                # mark the aux channel
                xmin = xmin + shift_amt
                ymin = ymin + shift_amt
                xmax = xmin + (width + width_amt)
                ymax = ymin + (height + height_amt)

                # handle overflow outside aux channel dimensions
                xmin = max(0, xmin)
                ymin = max(0, ymin)
                xmax = min(aux_width, xmax)
                ymax = min(aux_height, ymax)

            aux_channel[ymin:ymax, xmin:xmax] = 255
    return aux_channel


def draw_gt(xml_file_path, image):
    # load object coordinates from xml
    root = ElementTree.parse(xml_file_path).getroot()
    objects = root.findall("object")
    for obj in objects:
        if obj[0].text == "person":
            # bndbox = obj[4]
            bndbox = obj.find('bndbox')
            xmin = int(float(bndbox[0].text))
            ymin = int(float(bndbox[1].text))
            xmax = int(float(bndbox[2].text))
            ymax = int(float(bndbox[3].text))
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)
    return image


if __name__ == "__main__":
    # # Set a few configuration parameters.
    img_height = 300
    img_width = 300
    n_classes = 20  # other than background
    model_mode = 'inference'

    #
    # ####################################     Load Weights/MODEL here    ##########################################
    # # 1: Build the Keras model
    K.clear_session()  # Clear previous models from memory.

    model = ssd_300_aux(image_size=(img_height, img_width, 3),
                        n_classes=20,
                        mode='training',
                        l2_regularization=0.0005,
                        scales=[0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05],
                        # The scales for MS COCO are [0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05]
                        aspect_ratios_per_layer=[[1.0, 2.0, 0.5],
                                                 [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                                 [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                                 [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                                 [1.0, 2.0, 0.5],
                                                 [1.0, 2.0, 0.5]],
                        two_boxes_for_ar1=True,
                        steps=[8, 16, 32, 64, 100, 300],
                        offsets=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
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
    weights_path = './aux_input_trained_models/ssd300_pascal_07+12+PETS_no_rand_epoch-74_loss-2.3786_val_loss-2.2494.h5'
    # weights_path = './trained_models/ssd300_pascal_07+12_epoch-80_loss-2.7518_val_loss-2.5280.h5'

    model.load_weights(weights_path, by_name=True)

    # 3: Compile the model so that Keras won't complain the next time you load it.

    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

    ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)

    model.compile(optimizer=adam, loss=ssd_loss.compute_loss)

    ###############################################    DATA GENERATORS     ################################################

    # dataset = DataGenerator(load_images_into_memory=True, hdf5_dataset_path=None)

    # TODO: Set the paths to the dataset here.
    Pascal_VOC_dataset_images_dir = '../dataset/PASCAL_VOC/VOC2007/VOCdevkit/VOC2007/JPEGImages/'
    Pascal_VOC_dataset_annotations_dir = '../dataset/PASCAL_VOC/VOC2007/VOCdevkit/VOC2007/Annotations/'
    Pascal_VOC_dataset_image_set_filename = '../dataset/PASCAL_VOC/VOC2007/VOCdevkit/VOC2007/ImageSets/Main/test_subset_500.txt'

    output_dir = "temp_manual"
    # PETS_images_dir = '../dataset/model_training_data_pets/images/single_image_input/'
    # PETS_annotations_dir = '../dataset/model_training_data_pets/annotations/single_image_input/'
    # PETS_test_image_set_filename = '../dataset/model_training_data_pets/ImageSets/Main/test.txt'

    # read input images and predict
    test_file_ids = []
    with open(Pascal_VOC_dataset_image_set_filename) as f:
        test_file_ids = f.readlines()

    print(len(test_file_ids))
    print(test_file_ids[0])
    # for i in range(len(test_file_ids)):
    for i in range(120):
        image_path = "{}{}.jpg".format(Pascal_VOC_dataset_images_dir, test_file_ids[i][:-1])
        annot_path = "{}{}.xml".format(Pascal_VOC_dataset_annotations_dir, test_file_ids[i][:-1])
        print(image_path)
        # print(annot_path)

        input_images = []
        # img = Image.open(image_path)
        img = cv2.imread(image_path)
        assert img is not None
        # img = img.resize(size=(img_width, img_height))
        img = cv2.resize(img, dsize=(img_width, img_height))
        # cv2.imwrite("{}/test_image_{}.jpg".format(output_dir, i))
        # img = np.array(img, dtype=np.uint8)
        print("img shape : {}".format(img.shape))
        # assert image is not None
        # img = image.load_img(image_path, target_size=(img_height, img_width))
        # img = image.img_to_array(img)
        input_images.append(img)
        input_images = np.array(input_images)

        # create auxillary channel
        auxillary_channel = get_auxillary_channel(annot_path, randomize=False)
        aux_inputs = []
        aux_inputs.append(auxillary_channel)
        aux_inputs = np.array(aux_inputs)

        # dump aux_channel to file
        aux_ch_copy = auxillary_channel.copy()
        cv2.imwrite("{}/aux_channel_{}.jpg".format(output_dir, i), aux_ch_copy)
        # image.save("{}/aux_channel.jpg".format(output_dir))

        y_pred = model.predict([input_images, aux_inputs])

        # print("y_pred : {}".format(y_pred))
        # decode the predictions
        y_pred_decoded = decode_detections(y_pred,
                                           confidence_thresh=0.5,
                                           iou_threshold=0.4,
                                           top_k=200,
                                           normalize_coords=True,
                                           img_height=img_height,
                                           img_width=img_width)
        # print(y_pred_decoded)

        y_pred_decoded = y_pred_decoded[0].tolist()
        print(y_pred_decoded)

        # draw predicted boxes on the image
        # img = cv2.imread(image_path)
        for pred in y_pred_decoded:
            if int(pred[0]) == 15:
                xmin, ymin, xmax, ymax = int(pred[2]), int(pred[3]), int(pred[4]), int(pred[5])
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
        # # draw ground truth boxes
        # image = draw_gt(annot_path, img)

        # write image to file
        # cv2.imwrite("{}/{}.jpg".format(output_dir, test_file_ids[i][:-1]), img)
        cv2.imwrite("{}/test_img_{}.jpg".format(output_dir, i), img)
        # break

        # # load object coordinates from xml
        # root = ElementTree.parse(annot_path).getroot()
        # objects = root.findall("object")
        # for obj in objects:
        #     if obj[0].text == "person":
        #         # bndbox = obj[4]
        #         bndbox = obj.find('bndbox')
        #         xmin = int(float(bndbox[0].text))
        #         ymin = int(float(bndbox[1].text))
        #         xmax = int(float(bndbox[2].text))
        #         ymax = int(float(bndbox[3].text))
