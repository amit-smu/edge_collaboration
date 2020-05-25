"""test computed coordinate_mapping matrix"""

import pickle
import cv2
from xml.etree import ElementTree
import numpy as np
import random

if __name__ == "__main__":
    img_dir = "../../dataset/PETS/JPEGImages"
    annot_dir = "../../dataset/PETS/Annotations"

    frame_numbers = random.sample(range(0, 700), k=10)

    for frame_num in range(0, 30):
        # for frame_num in frame_numbers:
        # frame_num = 173
        frame_name1 = "frame_78_{:04d}".format(frame_num)
        img1 = cv2.imread("{}/{}.jpg".format(img_dir, frame_name1))
        annot_1 = "{}/{}.xml".format(annot_dir, frame_name1)

        frame_name2 = "frame_87_{:04d}".format(frame_num)
        img2 = cv2.imread("{}/{}.jpg".format(img_dir, frame_name2))
        annot_2 = "{}/{}.xml".format(annot_dir, frame_name2)

        # load coordinate_mapping here
        homography = None
        with open("homography_87_manual", "rb") as input_file:
            homography = pickle.load(file=input_file)

        obj_coords, obj_images = [], []

        # draw objects from ground truth
        root = ElementTree.parse(annot_1).getroot()
        objects = root.findall("object")
        for obj in objects:
            if obj[0].text == "person":
                # bndbox = obj[4]
                bndbox = obj.find('bndbox')
                xmin = int(float(bndbox[0].text))
                ymin = int(float(bndbox[1].text))
                xmax = int(float(bndbox[2].text))
                ymax = int(float(bndbox[3].text))

                # draw person rectangle
                cv2.rectangle(img1, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
        # cv2.imshow("image_1", img1)

        # draw boxes from collaborating camera
        root = ElementTree.parse(annot_2).getroot()
        objects = root.findall("object")
        for obj in objects:
            if obj[0].text == "person":
                # bndbox = obj[4]
                bndbox = obj.find('bndbox')
                xmin = int(float(bndbox[0].text))
                ymin = int(float(bndbox[1].text))
                xmax = int(float(bndbox[2].text))
                ymax = int(float(bndbox[3].text))

                cv2.rectangle(img2, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)

                # apply coordinate_mapping
                col_vector = np.array([xmin, ymin, 1])
                col_vector = col_vector.reshape((3, 1))
                col_vector_trans = np.dot(homography, col_vector)
                col_vector_trans = col_vector_trans.astype(dtype=np.int32)

                xmin1, ymin1, _ = col_vector_trans

                col_vector = np.array([xmax, ymax, 1])
                col_vector = col_vector.reshape((3, 1))
                col_vector_trans = np.dot(homography, col_vector)
                col_vector_trans = col_vector_trans.astype(dtype=np.int32)
                xmax1, ymax1, _ = col_vector_trans

                print("original points: {}".format([xmin, ymin, xmax, ymax]))
                print("translated points : {}\n".format([xmin1[0], ymin1[0], xmax1[0], ymax1[0]]))
                cv2.rectangle(img1, (xmin1[0], ymin1[0]), (xmax1[0], ymax1[0]), (0, 255, 0), 2)

        cv2.imshow("image_1", img1)
        cv2.imshow("image_2", img2)
        cv2.waitKey(-1)
