"""
test script for auxillary input models. Simply load model and draw the predictions on the test dataset
"""

import cv2
from keras import backend as K
from keras.optimizers import Adam
import numpy as np
from models.keras_ssd512 import ssd_512
from keras_loss_function.keras_ssd_loss import SSDLoss

import os
from xml.etree import ElementTree
import random

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def get_auxillary_channel(xml_file_path, randomize):
    aux_channel = np.full((img_height, img_width, 1), fill_value=0, dtype=np.uint8)

    SHIFT_RANGE = 50  # number of pixels to shift
    WIDTH_RANGE = 10
    HEIGHT_RANGE = 10
    aux_height, aux_width, _ = aux_channel.shape
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

            if randomize:
                width = xmax - xmin
                height = ymax - ymin

                # random perturbations
                shift_amt = random.randint(-1 * SHIFT_RANGE, SHIFT_RANGE)
                shift_amt_y = random.randint(-1 * SHIFT_RANGE, SHIFT_RANGE)
                width_amt = random.randint(-1 * WIDTH_RANGE, WIDTH_RANGE)
                height_amt = random.randint(-1 * HEIGHT_RANGE, HEIGHT_RANGE)
                # shift left or right
                # if flag == "LEFT":
                #     xmin = max(0, xmin - shift_amt)
                #     xmax = max(1, xmin + width)
                # elif flag == "RIGHT":
                #     xmin = min(aux_width, xmin + shift_amt) - 1
                #     xmax = min(aux_width, xmin + width)

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
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
    return image


if __name__ == "__main__":
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

    print(weights_path + "\n")

    model.load_weights(weights_path, by_name=True)

    # 3: Compile the model so that Keras won't complain the next time you load it.

    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

    ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)

    model.compile(optimizer=adam, loss=ssd_loss.compute_loss)

    ###############################################    DATA GENERATORS     ################################################

    # dataset = DataGenerator(load_images_into_memory=True, hdf5_dataset_path=None)

    output_dir = "temp"
    # PETS_images_dir = '../dataset/model_training_data_pets/images/single_image_input/'
    PETS_images_dir = '../dataset/compressed_data/knn_k_128/PETS_data/33%_left_sh_reg/512'
    PETS_annotations_dir = '../dataset/model_training_data_pets/annotations/single_image_input/'
    PETS_test_image_set_filename = '../dataset/model_training_data_pets/ImageSets/Main/test.txt'

    # WT_dataset_images_dir = "../dataset/Wildtrack_dataset/PNGImages_cropped_700x700"
    # WT_dataset_annotations_dir = "../dataset/Wildtrack_dataset/Annotations_cropped_700x700"
    # WT_dataset_test_image_set_filename = "../dataset/Wildtrack_dataset/ImageSets/Main/test_crop_700x700.txt"

    # WT_dataset_images_dir = "../dataset/Wildtrack_dataset/PNGImages"
    WT_dataset_images_dir = "../dataset/compressed_data/knn_k_128/WT_data/33%_left_sh_reg/512"
    WT_dataset_annotations_dir = "../dataset/Wildtrack_dataset/Annotations"
    WT_dataset_test_image_set_filename = "../dataset/Wildtrack_dataset/ImageSets/Main/test_30_cam_5.txt"

    # read input images and predict
    # DATASET = "PETS"
    DATASET = "WT"

    test_file_ids = []
    if DATASET == "WT":
        test_file_name = WT_dataset_test_image_set_filename
        images_dir = WT_dataset_images_dir
        annot_dir = WT_dataset_annotations_dir
        image_type = "png"
    elif DATASET == "PETS":
        test_file_name = PETS_test_image_set_filename
        images_dir = PETS_images_dir
        annot_dir = PETS_annotations_dir
        image_type = "jpg"

    # read test image names
    with open(test_file_name) as f:
        test_file_ids = f.read().split("\n")

    # print(len(test_file_ids))
    # print(test_file_ids[0])
    # for i in range(len(test_file_ids)):
    for file_id in test_file_ids:
        # for i in range(20):
        image_path = "{}/{}.{}".format(images_dir, file_id, image_type)
        annot_path = "{}/{}.xml".format(annot_dir, file_id)
        print(image_path)
        # print(annot_path)

        input_images = []
        # img = Image.open(image_path)
        img = cv2.imread(image_path)
        assert img is not None
        img_org = img.copy()

        # img = cv2.resize(img, dsize=(img_width // 2, img_height // 2))
        img = cv2.resize(img, dsize=(img_width, img_height))

        input_images.append(img)
        input_images = np.array(input_images)

        # create auxillary channel
        # auxillary_channel = get_auxillary_channel(annot_path, randomize=False)
        # aux_inputs = []
        # aux_inputs.append(auxillary_channel)
        # aux_inputs = np.array(aux_inputs)
        #
        # dump aux_channel to file
        # aux_ch_copy = auxillary_channel.copy()
        # cv2.imwrite("{}/aux_channel.jpg".format(output_dir), aux_ch_copy)
        # image.save("{}/aux_channel.jpg".format(output_dir))

        y_pred = model.predict(input_images)

        confidence_threshold = 0.5
        y_pred_thresh = [y_pred[k][y_pred[k, :, 1] > confidence_threshold] for k in range(y_pred.shape[0])]
        np.set_printoptions(precision=2, suppress=True, linewidth=90)
        # print("Predicted boxes:\n")
        # print('   class   conf xmin   ymin   xmax   ymax')
        # print(y_pred_thresh[0])

        # print("y_pred : {}".format(y_pred))
        # decode the predictions
        # y_pred_decoded = decode_detections(y_pred,
        #                                    confidence_thresh=0.5,
        #                                    iou_threshold=0.4,
        #                                    top_k=200,
        #                                    normalize_coords=True,
        #                                    img_height=img_height,
        #                                    img_width=img_width)
        # # print(y_pred_decoded)
        y_pred_decoded = y_pred_thresh
        y_pred_decoded = y_pred_decoded[0].tolist()
        # print(y_pred_decoded)

        # draw predicted boxes on the image
        # img = cv2.imread(image_path)
        org_img_height, org_img_width, _ = img_org.shape
        for pred in y_pred_decoded:
            if int(pred[0]) == 15:
                xmin, ymin, xmax, ymax = int(pred[2]), int(pred[3]), int(pred[4]), int(pred[5])

                xmin = int(xmin * org_img_width / img_width)
                ymin = int(ymin * org_img_height / img_height)
                xmax = int(xmax * org_img_width / img_width)
                ymax = int(ymax * org_img_height / img_height)
                cv2.rectangle(img_org, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
        # # draw ground truth boxes
        img = draw_gt(annot_path, img_org)

        # write image to file
        cv2.imwrite("{}/{}.{}".format(output_dir, file_id, image_type), img)

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
