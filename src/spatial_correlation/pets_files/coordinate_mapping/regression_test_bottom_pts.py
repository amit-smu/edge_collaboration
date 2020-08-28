import cv2
import os
import numpy as np
from xml.etree import ElementTree
import pickle

if __name__ == "__main__":
    DEBUG = False
    img_dir = "../../../dataset/PETS_1/JPEGImages"
    annot_dir = "../../../dataset/PETS_1/Annotations"

    src_cam = 57
    dst_cam = 78

    # coordinate_mapping = None
    # with open("homography_87_bottom_pts", "rb") as input_file:
    #     coordinate_mapping = pickle.load(input_file)
    #
    # print("coordinate_mapping : {}\n".format(coordinate_mapping))

    # load regressor
    regressor = None
    regressor_name = "regression_{}_bot_pts".format(src_cam)
    with open(regressor_name, 'rb') as input_f:
        regressor = pickle.load(input_f)
        assert regressor is not None
        print("regression model loaded.. :{}\n".format(regressor_name))

    for i in range(0, 795):
        src_img_name = "frame_{}_{:04d}.jpg".format(src_cam, i)
        src_img = cv2.imread("{}/{}".format(img_dir, src_img_name))
        src_annot_name = "frame_{}_{:04d}.xml".format(src_cam, i)
        src_annot_path = "{}/{}".format(annot_dir, src_annot_name)

        dst_img_name = "frame_{}_{:04d}.jpg".format(dst_cam, i)
        dst_img = cv2.imread("{}/{}".format(img_dir, dst_img_name))
        dst_annot_name = "frame_{}_{:04d}.xml".format(dst_cam, i)
        dst_annot_path = "{}/{}".format(annot_dir, dst_annot_name)

        src_root = ElementTree.parse(src_annot_path).getroot()
        src_objects = src_root.findall("object")

        for s_obj in src_objects:
            obj_id = s_obj.find("track_id").text
            bndbox = s_obj.find("bndbox")
            xmin = int(float(bndbox[0].text))
            ymin = int(float(bndbox[1].text))
            xmax = int(float(bndbox[2].text))
            ymax = int(float(bndbox[3].text))

            width = xmax - xmin
            mid_x = int(xmin + width / 2)

            cv2.circle(src_img, (mid_x, ymax), 4, (0, 0, 255), 2)
            cv2.putText(src_img, "{}".format(obj_id), (mid_x, ymax - 3), cv2.FONT_HERSHEY_PLAIN, 1.2, (255, 0, 0), 1)

            # translate points to dst imgage

            mid_x_t, ymax_t = regressor.predict([[mid_x, ymax]])[0]
            mid_x_t = int(mid_x_t)
            ymax_t = int(ymax_t)

            # compute bounding box
            org_height = ymax - ymin
            org_width = (xmax - xmin)
            bb_xmin = int(mid_x_t - org_width / 2)
            bb_xmax = int(mid_x_t + org_width / 2)
            bb_ymin = ymax_t - org_height
            bb_ymax = ymax_t

            cv2.circle(dst_img, (mid_x_t, ymax_t), 4, (0, 255, 0), 2)
            cv2.putText(dst_img, "{}".format(obj_id), (mid_x_t, ymax_t - 3), cv2.FONT_HERSHEY_PLAIN, 1.2,
                        (0, 255, 0), 1)
            cv2.rectangle(dst_img, (bb_xmin, bb_ymin), (bb_xmax, bb_ymax), (0, 255, 255), 1)
            cv2.putText(dst_img, "{}".format(obj_id), (bb_xmin, bb_ymin - 3), cv2.FONT_HERSHEY_PLAIN, 1.2,
                        (0, 255, 0), 1)
            # col_vector = np.array([mid_x, ymax, 1])
            # col_vector = col_vector.reshape((3, 1))
            # # homography_inv = np.linalg.inv(coordinate_mapping)
            # col_vector_trans = np.dot(coordinate_mapping, col_vector)
            # col_vector_trans = col_vector_trans.astype(dtype=np.int32)
            # xmin1, ymax1, _ = col_vector_trans
            # print(xmin1[0], ymax1[0])
            # cv2.circle(dst_img, (xmin1[0], ymax1[0] - 1), 4, (0, 255, 0), 2)
            # cv2.putText(dst_img, "{}".format(obj_id), (xmin1[0], ymax1[0] - 3), cv2.FONT_HERSHEY_PLAIN, 1.2,
            #             (0, 255, 1), 1)

        cv2.imshow("src_img", src_img)
        cv2.imshow("dst_img", dst_img)
        cv2.waitKey(-1)
