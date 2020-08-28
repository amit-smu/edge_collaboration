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
import os
import matplotlib.pyplot as plt


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

    # frame_num = 0
    f_names_1, f_names_2 = [], []
    f_annot_1, f_annot_2 = [], []

    # load svm model
    model_file = open("svm_model_8_samples(82_82).sav", 'rb')
    classifier = pickle.load(model_file)

    src_points, dst_points = [], []

    frame_numbers = sorted(random.sample(range(0, 700), k=10))

    # for i in range(0, 15):
    for frame_num in frame_numbers:
        print("frame_num : {}".format(frame_num))
        obj_coords_1, obj_coords_2 = [], []
        obj_images_1, obj_images_2 = [], []
        obj_embeddings_1, obj_embeddings_2 = [], []

        frame_name1 = "frame_78_{:04d}".format(frame_num)
        img1 = cv2.imread("{}/{}.jpg".format(img_dir, frame_name1))
        annot_1 = "{}/{}.xml".format(annot_dir, frame_name1)

        frame_name2 = "frame_87_{:04d}".format(frame_num)
        img2 = cv2.imread("{}/{}.jpg".format(img_dir, frame_name2))
        annot_2 = "{}/{}.xml".format(annot_dir, frame_name2)

        if DEBUG:
            cv2.imshow("img1", img1)
            cv2.imshow("img2", img2)

        # get object coords in both frames

        if (not os.path.exists(annot_1)) or (not os.path.exists(annot_2)):
            print("annotation file doesn't exist\n")
            continue
        obj_coords, obj_images = get_obj_coords(xml_path=annot_1, image=img1)
        obj_coords_1.extend(obj_coords)
        obj_images_1.extend(obj_images)

        obj_coords, obj_images = get_obj_coords(xml_path=annot_2, image=img2)
        obj_coords_2.extend(obj_coords)
        obj_images_2.extend(obj_images)

        # extract embeddings for each object
        for obj_img in obj_images_1:
            embedding = utils.get_obj_embedding(obj_img, temp_dir_path="../temp_files")
            obj_embeddings_1.append(embedding)

        for obj_img in obj_images_2:
            embedding = utils.get_obj_embedding(obj_img, temp_dir_path="../temp_files")
            obj_embeddings_2.append(embedding)

        # match objects using SVN classifier
        for v1, embd1 in enumerate(obj_embeddings_1):
            cat_1 = classifier.predict([embd1])
            for v2, embd2 in enumerate(obj_embeddings_2):
                cat_2 = classifier.predict([embd2])
                if cat_1 == cat_2:
                    # add points to the src and destination point lists
                    if DEBUG:
                        cv2.imshow("obj1", obj_images_1[v1])
                        cv2.imshow("obj2", obj_images_2[v2])
                        cv2.waitKey(-1)
                        cv2.destroyAllWindows()
                    src_points.append(obj_coords_1[v1])
                    dst_points.append(obj_coords_2[v2])
                    obj_embeddings_2.remove(embd2)
                    obj_coords_2.remove(obj_coords_2[v2])
                    obj_images_2.remove(obj_images_2[v2])
                    break

        # convert points in src and dst lists to four points (instead of one array of points)
        src_points_1, dst_points_1 = [], []
        for pt in src_points:
            xmin, ymin, xmax, ymax = pt
            src_points_1.append([xmin, ymin])
            # src_points_1.append([xmax, ymin])
            src_points_1.append([xmax, ymax])
            # src_points_1.append([xmin, ymax])
        src_points_1 = np.array(src_points_1)

        for pt in dst_points:
            xmin, ymin, xmax, ymax = pt
            dst_points_1.append([xmin, ymin])
            # dst_points_1.append([xmax, ymin])
            dst_points_1.append([xmax, ymax])
            # dst_points_1.append([xmin, ymax])
        dst_points_1 = np.array(dst_points_1)

    # compute coordinate_mapping
    homography, status = cv2.findHomography(srcPoints=dst_points_1, dstPoints=src_points_1)
    print(homography)
    # if status:
    with open("homography_7_8", "wb") as output:
        pickle.dump(homography, file=output)

    homography = None
    with open("homography_7_8", "rb") as input_file:
        homography = pickle.load(file=input_file)
    print("\n")
    print(homography)

    # scatter plot of poitns
    plt.plot()
