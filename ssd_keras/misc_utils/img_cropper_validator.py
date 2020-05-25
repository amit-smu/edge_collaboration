"""
module to check if the image cropper has cropped properly or not
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
        # test_file_path = r"../../dataset/Wildtrack_dataset/ImageSets/Main/test.txt"
        imageset_dir = r"../../dataset/Wildtrack_dataset/PNGImages_cropped_700x700"
        annotation_dir = r"../../dataset/Wildtrack_dataset/Annotations_cropped_700x700"
        img_type = "png"

    if DATASET == "PETS":
        org_img_w, org_img_h = (720, 576)
        # test_file_path = r"../../dataset/PETS_org/ImageSets/Main/test_12.txt"
        imageset_dir = r"../../dataset/PETS_org/JPEGImages_cropped_{}x{}".format(
            dsize[0], dsize[1])
        annotation_dir = r"../../dataset/PETS_org/Annotations_cropped_{}x{}".format(
            dsize[0], dsize[1])
        img_type = "jpg"

    image_file_names = os.listdir("{}".format(imageset_dir))
    print("Total images : {}\n".format(len(image_file_names)))

    for img_name in image_file_names:
        # imageset_dir = r"../../dataset/Wildtrack_dataset/PNGImages"
        # annotation_dir = r"../../dataset/Wildtrack_dataset/Annotations"
        # img_name = "C5_00001395.png"
        image = cv2.imread("{}/{}".format(imageset_dir, img_name))
        assert image is not None

        annot_name = "{}.xml".format(img_name[:-4])
        annot_file_path = "{}/{}".format(annotation_dir, annot_name)

        root = ElementTree.parse(annot_file_path).getroot()
        assert root is not None
        objects = root.findall("object")
        for obj in objects:
            bndbox = obj.find("bndbox")
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)

            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

        cv2.imshow("{}".format(img_name), image)
        cv2.waitKey(-1)
        cv2.destroyAllWindows()
