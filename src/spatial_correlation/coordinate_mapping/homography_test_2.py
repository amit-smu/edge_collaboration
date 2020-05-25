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

    for i in range(0, 30):
        frame_num = 790
        # for frame_num in frame_numbers:
        # frame_num = 173
        dst_frame_name = "frame_78_{:04d}".format(frame_num + i)
        dst_img = cv2.imread("{}/{}.jpg".format(img_dir, dst_frame_name))
        dst_annot = "{}/{}.xml".format(annot_dir, dst_frame_name)

        src_frame_name = "frame_87_{:04d}".format(frame_num + i)
        src_img = cv2.imread("{}/{}.jpg".format(img_dir, src_frame_name))
        src_annot = "{}/{}.xml".format(annot_dir, src_frame_name)

        # load coordinate_mapping here
        homography = None
        with open("homography_87_manual", "rb") as input_file:
            homography = pickle.load(file=input_file)

        print("coordinate_mapping : \n{}".format(homography))

        obj_coords, obj_images = [], []

        # map boxes from collaborating camera (source) to reference cam (destination)
        root = ElementTree.parse(src_annot).getroot()
        objects = root.findall("object")
        for obj in objects:
            if obj[0].text == "person":
                # bndbox = obj[4]
                bndbox = obj.find('bndbox')
                xmin = int(float(bndbox[0].text))
                ymin = int(float(bndbox[1].text))
                xmax = int(float(bndbox[2].text))
                ymax = int(float(bndbox[3].text))

                cv2.rectangle(src_img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
                # put id on boxes
                r_id = random.randint(0, 1000)
                cv2.putText(src_img, "{}".format(r_id), (xmin, ymax + 5), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 1)

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

                # print("original points: {}".format([xmin, ymin, xmax, ymax]))
                # print("translated points : {}\n".format([xmin1[0], ymin1[0], xmax1[0], ymax1[0]]))
                cv2.rectangle(dst_img, (xmin1[0], ymin1[0]), (xmax1[0], ymax1[0]), (0, 255, 0), 2)
                cv2.putText(dst_img, "{}".format(r_id), (xmin1[0], ymax1[0] + 5), cv2.FONT_HERSHEY_PLAIN, 1.5,
                            (0, 255, 0), 1)

        cv2.imshow("src_image", src_img)
        cv2.imshow("dst_image", dst_img)
        cv2.waitKey(-1)
