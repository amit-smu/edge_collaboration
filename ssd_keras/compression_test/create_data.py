"""
module to create data for image compression vs obj detection experiments of CollabCam
"""

import numpy as np
import shutil
import os
import cv2

if __name__ == "__main__":
    DIR_PATH = ""
    # TEST_DATASET = "PETS"
    TEST_DATASET = "WT"
    SHARED_AREA_RES = 96
    SHARED_AREA_COORDS = [0, 0, 1, 1]  # in full image coordinates

    output_dir_path = "./{}_data/33%_left_sh_reg/{}".format(TEST_DATASET, SHARED_AREA_RES)
    #output_dir_path = "./{}_data/0%_sh_reg/{}".format(TEST_DATASET, SHARED_AREA_RES)
    print("output_dir_path: {}".format(output_dir_path))

    PETS_images_dir = '../../dataset/PETS_org/JPEGImages'
    PETS_annotations_dir = '../dataset/PETS_org/Annotations'
    PETS_test_image_set_filename = '../../dataset/PETS_org/ImageSets/Main/test_30_cam_8.txt'

    WT_images_dir = "../../dataset/Wildtrack_dataset/PNGImages"
    WT_annotations_dir = "../dataset/Wildtrack_dataset/Annotations"
    WT_test_image_set_filename = "../../dataset/Wildtrack_dataset/ImageSets/Main/test_30_cam_5.txt"

    # create target directory
    if not os.path.exists(output_dir_path):
        print("creating directory : {}".format(output_dir_path))
        os.makedirs(output_dir_path)

    # read test images
    if TEST_DATASET == "PETS":
        images_dirs = PETS_images_dir
        image_set_filenames = PETS_test_image_set_filename
        image_ext = ".jpg"
    elif TEST_DATASET == "WT":
        images_dirs = WT_images_dir
        image_set_filenames = WT_test_image_set_filename
        image_ext = ".png"

    file_names = None
    with open(image_set_filenames) as t:
        file_names = t.read().split("\n")     
#        print(file_names)
#        print(type(file_names))
        
    for f_name in file_names:
        f_name = "{}{}".format(f_name, image_ext)
        print("f_name: {}".format(f_name))
        image = cv2.imread("{}/{}".format(images_dirs, f_name))
        assert image is not None
        image_resized = cv2.resize(image, dsize=(512, 512), interpolation=cv2.INTER_AREA)

        # create mixed res image
        xmin_org, ymin_org, xmax_org, ymax_org = SHARED_AREA_COORDS  # in org cam resolution (1920x1080 for WT)
        if TEST_DATASET == "PETS":
            xmin_tr = int((xmin_org / 720.0) * 512)
            ymin_tr = int((ymin_org / 576.0) * 512)
            xmax_tr = int((xmax_org / 720.0) * 512)
            ymax_tr = int((ymax_org / 576.0) * 512)
        elif TEST_DATASET == "WT":
            xmin_tr = int((xmin_org / 1920.0) * 512)
            ymin_tr = int((ymin_org / 1080.0) * 512)
            xmax_tr = int((xmax_org / 1920.0) * 512)
            ymax_tr = int((ymax_org / 1080.0) * 512)

        reg_width = xmax_tr - xmin_tr  # width of shared region in 512x512 image
        reg_height = ymax_tr - ymin_tr
        reg_width_tr = int((reg_width / 512.0) * SHARED_AREA_RES)  # new width as per 224x224 overall resolution
        reg_height_tr = int((reg_height / 512.0) * SHARED_AREA_RES)
        shared_reg_target_res = (reg_width_tr, reg_height_tr)  # shared area res as per 224x224 overall img res

        shared_reg = image_resized[ymin_tr:ymax_tr, xmin_tr:xmax_tr]
        temp = cv2.resize(shared_reg, dsize=shared_reg_target_res, interpolation=cv2.INTER_AREA)
        shared_reg = cv2.resize(temp, dsize=(reg_width, reg_height), interpolation=cv2.INTER_CUBIC)
        image_resized[ymin_tr:ymax_tr, xmin_tr:xmax_tr] = shared_reg

        # write mixed-res image to disk
        cv2.imwrite("{}/{}".format(output_dir_path, f_name), image_resized)
