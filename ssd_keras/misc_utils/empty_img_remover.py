"""
identify images from cropped images, for which not a single bounding box of ground turth was included while cropping
"""

import cv2
import numpy as np
import os
from xml.etree import ElementTree

if __name__ == "__main__":
    dsize = (700, 700)
    DATASET = "WT"
    # DATASET = "PETS"

    print("Dsize : {}".format(dsize))
    print("Dataset : {}".format(DATASET))

    if DATASET == "WT":
        org_img_w, org_img_h = (1920, 1080)
        test_file_path = r"../../dataset/Wildtrack_dataset/ImageSets/Main/test.txt"
        imageset_dir = r"../../dataset/Wildtrack_dataset/PNGImages_cropped_{}x{}".format(
            dsize[0], dsize[1])
        annotation_dir = r"../../dataset/Wildtrack_dataset/Annotations_cropped_{}x{}".format(
            dsize[0], dsize[1])
        img_type = "png"

    if DATASET == "PETS":
        org_img_w, org_img_h = (720, 576)
        test_file_path = r"../../dataset/PETS_org/ImageSets/Main/test_12.txt"
        imageset_dir = r"../../dataset/PETS_org/JPEGImages_cropped_{}x{}".format(
            dsize[0], dsize[1])
        annotation_dir = r"../../dataset/PETS_org/Annotations_cropped_{}x{}".format(
            dsize[0], dsize[1])
        img_type = "jpg"

    image_file_names = os.listdir("{}".format(imageset_dir))
    print("Total images : {}\n".format(len(image_file_names)))
