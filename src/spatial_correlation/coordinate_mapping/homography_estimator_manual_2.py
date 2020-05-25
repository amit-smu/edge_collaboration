"""
module to estimate homogrpahy between two camera views. Using static points as reference
"""

import cv2
import os
import pickle
from xml.etree import ElementTree
import utils
import numpy as np
import random


def get_obj_coords(xml_path, image):
    obj_coords = []

    # load object coordinates from xml
    root = ElementTree.parse(xml_path).getroot()
    objects = root.findall("object")
    # sort by name
    objects_sorted = sorted(objects, key=lambda obj: obj.find('name').text)
    for obj in objects_sorted:
        # bndbox = obj[4]
        obj_name = obj.find('name').text

        if obj_name == "lady":
            continue

        bndbox = obj.find('bndbox')
        xmin = int(float(bndbox[0].text))
        ymin = int(float(bndbox[1].text))
        # if obj_name == "lady":
        xmax = int(float(bndbox[2].text))
        ymax = int(float(bndbox[3].text))

        # add points to list
        obj_coords.append([xmin, ymin])
        # obj_coords.append([xmax, ymin])
        # obj_coords.append([xmax, ymax])
        # obj_coords.append([xmin, ymax])
        # center_x = int((xmin + xmax) / 2)
        # center_y = int((ymin + ymax) / 2)
        # obj_coords.append([center_x, center_y])

    return obj_coords


if __name__ == "__main__":
    DEBUG = False
    img_dir = "../../dataset/PETS/JPEGImages"
    annot_dir = "../../dataset/PETS/Homography"

    FLAG = "87"
    objects_87 = ["center", 'corner_a', 'corner_b', "lady", 'mark', 'mark_on_line']

    obj_list = []

    if FLAG == "87":
        obj_list = objects_87

    src_frame_name = "frame_87_0793.jpg"
    dst_frame_name = "frame_78_0793.jpg"

    src_coords, dst_coords = [], []

    src_img = cv2.imread("{}/{}".format(img_dir, src_frame_name))
    src_annot = "{}/{}.xml".format(annot_dir, src_frame_name[:-4])

    dst_img = cv2.imread("{}/{}".format(img_dir, dst_frame_name))
    dst_annot = "{}/{}.xml".format(annot_dir, dst_frame_name[:-4])

    # get object coordinates from xml
    src_points = get_obj_coords(xml_path=src_annot, image=src_img)
    dst_points = get_obj_coords(xml_path=dst_annot, image=dst_img)

    print("src_poits : {}".format(src_points))
    print("\ndst_poits : {}".format(dst_points))
    # convert points in src and dst lists to four points (instead of one array of points)

# compute coordinate_mapping
src_points_1 = np.array(src_points, dtype=np.float)
dst_points_1 = np.array(dst_points, dtype=np.float)
print("src_points :{}".format(src_points_1))
print("\ndst_points :{}".format(dst_points_1))

homography, status = cv2.findHomography(srcPoints=src_points_1, dstPoints=dst_points_1, method=cv2.RANSAC)
print(homography)

# if status:
with open("homography_87_manual", "wb") as output:
    pickle.dump(homography, file=output)

homography = None
with open("homography_87_manual", "rb") as input_file:
    homography = pickle.load(file=input_file)

print("\ncoordinate_mapping :\n {}\n".format(homography))
