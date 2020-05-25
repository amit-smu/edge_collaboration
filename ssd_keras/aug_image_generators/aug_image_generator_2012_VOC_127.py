"""module to generate augmented images i.e. 4 channel images where 4th chanell is
bounding boxes from another camera
"""

import numpy as np
import cv2
import os
import xml.dom.minidom
from xml.etree import ElementTree
import random
from PIL import Image
import sys


def gen_random_perturbs(xmin, ymin, xmax, ymax, aux_channel):
    aux_width, aux_height, _ = aux_channel.shape
    width = xmax - xmin
    height = ymax - ymin

    # box 1 (upper left)
    rand_x = random.randint(xmin - RANDOM_X, xmin + RANDOM_X)
    rand_y = random.randint(ymin - RANDOM_Y, ymin + RANDOM_Y)

    xmin1 = max(0, rand_x)
    ymin1 = max(0, rand_y)
    xmax1 = xmin + width
    ymax1 = ymin + height
    aux_channel[ymin1:ymax1, xmin1:xmax1] = 255
    if DEBUG:
        cv2.imshow("aux_after", aux_channel)
        cv2.waitKey(-1)

    # box 2 (lower left)
    rand_x = random.randint(xmin - RANDOM_X, xmin + RANDOM_X)
    rand_y = random.randint(ymax - RANDOM_Y, ymax + RANDOM_Y)
    xmin1 = max(0, rand_x)
    ymin1 = rand_y
    xmax1 = xmin1 + width
    ymax1 = min(ymin1 + height, aux_height)
    aux_channel[ymin1:ymax1, xmin1:xmax1] = 255
    if DEBUG:
        cv2.imshow("aux_after", aux_channel)
        cv2.waitKey(-1)

    # box 3 (lower right)
    rand_x = random.randint(xmax - RANDOM_X, xmax + RANDOM_X)
    rand_y = random.randint(ymax - RANDOM_Y, ymax + RANDOM_Y)
    xmin1 = rand_x
    ymin1 = rand_y
    xmax1 = min(xmin1 + width, aux_width)
    ymax1 = min(ymin1 + height, aux_height)
    aux_channel[ymin1:ymax1, xmin1:xmax1] = 255
    if DEBUG:
        cv2.imshow("aux_after", aux_channel)
        cv2.waitKey(-1)

    # box 4 (upper right)
    rand_x = random.randint(xmax - RANDOM_X, xmax + RANDOM_X)
    rand_y = random.randint(ymin - RANDOM_Y, ymin + RANDOM_Y)
    xmin1 = rand_x
    ymin1 = max(0, rand_y)
    xmax1 = min(xmin1 + width, aux_width)
    ymax1 = ymin1 + height
    aux_channel[ymin1:ymax1, xmin1:xmax1] = 255
    if DEBUG:
        cv2.imshow("aux_after", aux_channel)
        cv2.waitKey(-1)

    return aux_channel


if __name__ == "__main__":
    DEBUG = True
    RANDOM_X = 25
    RANDOM_Y = 15

    dataset_dir = "../dataset"
    # image_dir = "../dataset/PASCAL_VOC/VOC2007/VOCdevkit/VOC2007/JPEGImages"
    image_dir = "{}/PASCAL_VOC/VOC2012/VOCdevkit/VOC2012/JPEGImages".format(dataset_dir)
    # annot_dir = "../dataset/PASCAL_VOC/VOC2007/VOCdevkit/VOC2007/Annotations"
    annot_dir = "{}/PASCAL_VOC/VOC2012/VOCdevkit/VOC2012/Annotations".format(dataset_dir)

    # output_dir = "../dataset/PASCAL_VOC/VOC2007/VOCdevkit/VOC2007/JPEGImages_aux_127"
    output_dir = "{}/PASCAL_VOC/VOC2012/VOCdevkit/VOC2012/JPEGImages_aux_127".format(dataset_dir)

    # read images in directory
    images = os.listdir(image_dir)
    print(len(images))
    # sys.exit(-1)
    skip_images = ["2007", "2009", "2009", "2010"]
    for img_name in images:
        # read images
        # img_name = "001362.jpg"

        print(img_name)

        ############################################################################
        ####################### TEMPORARY CODE #####################################
        ############################################################################
        # year = int(img_name.split("_")[0])
        # if year < 2011:
        #     continue
        # image_name_number = int(img_name.strip(".jpg").split("_")[1])
        # if image_name_number < 3350:
        #     continue
        img_name_1 = "{}/{}.png".format(output_dir, img_name.strip(".jpg"))
        if os.path.exists(img_name_1):
            continue
        ############################################################################

        img_path = "{}/{}".format(image_dir, img_name)
        # image = cv2.imread(img_path)
        image = Image.open(img_path)
        image = np.array(image, dtype=np.uint8)
        # assert image is not None
        height, width, channels = image.shape

        # aux_org = aux_channel.copy()
        # cv2.imshow("aux", image)
        # cv2.waitKey(-1)
        # if DEBUG:
        #     cv2.imshow("aux_org", aux_org)

        # get annotation file
        annotation_path = "{}/{}.xml".format(annot_dir, img_name.strip(".jpg"))
        # dom_tree = xml.dom.minidom.parse(annotation_path)
        # collection = dom_tree.documentElement
        # objects = collection.getElementsByTagName("object")
        # for obj in objects:
        #     class_name = obj.childNodes[1].firstChild.nodeValue
        #     if class_name == "person":
        #         print()
        root = ElementTree.parse(annotation_path).getroot()
        objects = root.findall("object")

        if len(objects) > 0:
            aux_channel = np.full((height, width, 1), 255, dtype=np.uint8)
        else:
            aux_channel = np.full((height, width, 1), 255, dtype=np.uint8)

        for obj in objects:
            if obj[0].text == "person":
                # bndbox = obj[4]
                bndbox = obj.find('bndbox')
                xmin = bndbox[0].text
                ymin = bndbox[1].text
                xmax = bndbox[2].text
                ymax = bndbox[3].text

                xmin = int(float(xmin))
                ymin = int(float(ymin))
                xmax = int(float(xmax))
                ymax = int(float(ymax))

                # xmin, ymin, xmax, ymax = int(bndbox[0].text), int(bndbox[1].text), int(bndbox[2].text), int(
                #     bndbox[3].text)
                # mark the bounding box coords in aux
                aux_channel[ymin:ymax, xmin:xmax] = 127
                # if DEBUG:
                #     cv2.imshow("aux_org", aux_org)
                # aux_channel = gen_random_perturbs(xmin, ymin, xmax, ymax, aux_channel)
        # if DEBUG:
        #     cv2.imshow("aux_org", aux_org)
        #     # cv2.imshow("aux_after", aux_channel)
        #     cv2.imshow("image", image)
        #     cv2.waitKey(-1)
        #     cv2.destroyAllWindows()
        # append aux channel to image
        image_aux = np.concatenate((image, aux_channel), axis=2)
        aug_image = Image.fromarray(image_aux)
        aug_image.save("{}/{}.png".format(output_dir, img_name.strip(".jpg")), format="PNG")
        # img = Image.open("{}/{}.png".format(output_dir, img_name.strip(".jpg")))
        # img.show()
        # cv2.imwrite("{}/{}".format(output_dir, img_name), image_aux)
