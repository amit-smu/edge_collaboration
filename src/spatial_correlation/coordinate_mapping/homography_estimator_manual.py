"""
module to estimate homogrpahy between two camera views
"""

import cv2
import os
import pickle
from xml.etree import ElementTree
import utils
import numpy as np
import random


def get_obj_coords(xml_path, image):
    obj_coords, obj_images = [], []
    # load object coordinates from xml
    root = ElementTree.parse(xml_path).getroot()
    objects = root.findall("object")
    for obj in objects:
        if obj[0].text == "person":
            # bndbox = obj[4]
            bndbox = obj.find('bndbox')
            xmin = int(float(bndbox[0].text))
            ymin = int(float(bndbox[1].text))
            xmax = int(float(bndbox[2].text))
            ymax = int(float(bndbox[3].text))

            # obj_coords.append([(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)])
            obj_coords.append([xmin, ymin, xmax, ymax])
            sub_img = image[ymin:ymax, xmin:xmax]
            # if DEBUG:
            #     cv2.imshow("{}".format(xmin + xmax), sub_img)
            obj_images.append(sub_img)
    # if DEBUG:
    #     cv2.waitKey(-1)
    return obj_coords, obj_images


if __name__ == "__main__":
    DEBUG = False
    img_dir = "../../dataset/PETS/JPEGImages"
    annot_dir = "../../dataset/PETS/Annotations"

    # frame_num = 10
    f_names_1, f_names_2 = [], []
    f_annot_1, f_annot_2 = [], []

    src_points, dst_points = [], []
    src_points_1, dst_points_1 = [], []
    frame_numbers = random.sample(range(0, 700), k=4)

    # for i in range(0, 15):
    for frame_num in frame_numbers:
        # for i in range(10, 15):
        obj_coords_1, obj_coords_2 = [], []
        obj_images_1, obj_images_2 = [], []
        obj_embeddings_1, obj_embeddings_2 = [], []

        frame_name1 = "frame_78_{:04d}".format(frame_num)
        img1 = cv2.imread("{}/{}.jpg".format(img_dir, frame_name1))
        annot_1 = "{}/{}.xml".format(annot_dir, frame_name1)

        frame_name2 = "frame_67_{:04d}".format(frame_num)
        img2 = cv2.imread("{}/{}.jpg".format(img_dir, frame_name2))
        annot_2 = "{}/{}.xml".format(annot_dir, frame_name2)

        # if DEBUG:
        #     cv2.imshow("img1", img1)
        #     cv2.imshow("img2", img2)
        # cv2.imshow("img1", img1)
        # cv2.imshow("img2", img2)
        # cv2.waitKey(-1)

        # get object coords in both frames
        obj_coords, obj_images = get_obj_coords(xml_path=annot_1, image=img1)
        obj_coords_1.extend(obj_coords)
        obj_images_1.extend(obj_images)

        obj_coords, obj_images = get_obj_coords(xml_path=annot_2, image=img2)
        obj_coords_2.extend(obj_coords)
        obj_images_2.extend(obj_images)

        # extract embeddings for each object
        # for obj_img in obj_images_1:
        #     embedding = utils.get_obj_embedding(obj_img, temp_dir_path="../temp_files")
        #     obj_embeddings_1.append(embedding)
        #
        # for obj_img in obj_images_2:
        #     embedding = utils.get_obj_embedding(obj_img, temp_dir_path="../temp_files")
        #     obj_embeddings_2.append(embedding)

        # match objects using SVN classifier
        for v1, obj1 in enumerate(obj_coords_1):
            img1_copy = img1.copy()
            cv2.rectangle(img1_copy, (obj1[0], obj1[1]), (obj1[2], obj1[3]), (0, 255, 0), 2)
            cv2.imshow("image_1_{}".format(v1), img1_copy)
            cv2.moveWindow("image_1_{}".format(v1), 100, 100)
            # cv2.imshow("obj1", obj1)

            for v2, obj2 in enumerate(obj_coords_2):
                # cv2.imshow("obj2", obj2)
                # cv2.waitKey(1000)
                img2_copy = img2.copy()
                cv2.rectangle(img2_copy, (obj2[0], obj2[1]), (obj2[2], obj2[3]), (0, 255, 0), 2)
                cv2.imshow("image_2_{}".format(v2), img2_copy)
                cv2.moveWindow("image_2_{}".format(v2), 600, 600)
                cv2.waitKey(500)
                obj_same = input("Are objects same?")
                print(obj_same)
                cv2.destroyWindow("image_2_{}".format(v2))

                if obj_same == "yes" or obj_same == "Yes":
                    # add points to the src and destination point lists
                    s_point = obj2
                    src_points.append(s_point)
                    d_point = obj1
                    dst_points.append(d_point)
                    print("s_point: {}, d_point : {}\n".format(s_point, d_point))
                    obj_coords_2.remove(obj2)
                    # obj_coords_2.remove(obj_coords_2[v2])
                    cv2.destroyWindow("image_1_{}".format(v1))
                    break
            cv2.destroyAllWindows()
            # cv2.destroyWindow("obj1")

        print("src_poits : {}".format(src_points))
        print("\ndst_poits : {}".format(dst_points))
        # convert points in src and dst lists to four points (instead of one array of points)

        for pt in src_points:
            xmin, ymin, xmax, ymax = pt
            src_points_1.append([xmin, ymin])
            # src_points_1.append([xmax, ymin])
            src_points_1.append([xmax, ymax])
            # src_points_1.append([xmin, ymax])

        for pt in dst_points:
            xmin, ymin, xmax, ymax = pt
            dst_points_1.append([xmin, ymin])
            # dst_points_1.append([xmax, ymin])
            dst_points_1.append([xmax, ymax])
            # dst_points_1.append([xmin, ymax])

    # compute coordinate_mapping
    src_points_1 = np.array(src_points_1, dtype=np.float)
    dst_points_1 = np.array(dst_points_1, dtype=np.float)
    print("src_points :{}".format(src_points_1))
    print("\ndst_points :{}".format(dst_points_1))
    homography, status = cv2.findHomography(srcPoints=src_points_1, dstPoints=dst_points_1)
    print(homography)
    # if status:
    with open("homography_6_7_manual_1", "wb") as output:
        pickle.dump(homography, file=output)

    homography = None
    with open("homography_6_7_manual_1", "rb") as input_file:
        homography = pickle.load(file=input_file)

    print(homography)
