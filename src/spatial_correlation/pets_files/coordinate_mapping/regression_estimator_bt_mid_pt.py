"""
regression estimator for coordinate mapping from one frame to another. Linear regression
use bottom points of each bounding box. These points are on the road
"""

import cv2
import os
import numpy as np
from xml.etree import ElementTree
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd


def fit_linear_regression_bt_pts(src_points, dst_points):
    """
    fit a multivariate linear regression. data only contains bottom-mid points of bounding boxes
    :param src_points:
    :param dst_points:
    :return:
    """
    regressor = LinearRegression()

    # add data to df
    points = []
    for i in range(len(src_points)):
        points.append(src_points[i] + dst_points[i])

    df = pd.DataFrame(data=points, columns=['src_bot_x', 'src_bot_y', 'dst_bot_x', 'dst_bot_y'])
    df = df.astype('int32')
    X = df[['src_bot_x', 'src_bot_y']].values
    Y = df[['dst_bot_x', 'dst_bot_y']].values

    # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=100)
    regressor.fit(X=X, y=Y)
    print(regressor.coef_)

    # save the model to file
    with open("regression_{}_bot_pts".format(src_cam), "wb") as output:
        pickle.dump(regressor, output)
        print("Saved model : regression_{}_bot_pts\n".format(src_cam))
    print(regressor.predict([[120, 150]]))

    regressor = None
    with open("regression_{}_bot_pts".format(src_cam), "rb") as input_file:
        regressor = pickle.load(input_file)
    print(regressor.predict([[120, 150]]))
    print()


if __name__ == "__main__":
    DEBUG = True
    img_dir = "../../../dataset/PETS_1/JPEGImages"
    annot_dir = "../../../dataset/PETS_1/Annotations"

    # ref_cam = 7
    # collab_cams = [8,6,5]
    src_cam = 57
    dst_cam = 78

    src_points, dst_points = [], []

    for i in range(0, 750):
        src_annot_name = "frame_{}_{:04d}.xml".format(src_cam, i)
        src_annot_path = "{}/{}".format(annot_dir, src_annot_name)

        dst_annot_name = "frame_{}_{:04d}.xml".format(dst_cam, i)
        dst_annot_path = "{}/{}".format(annot_dir, dst_annot_name)

        if not (os.path.exists(src_annot_path) and os.path.exists(dst_annot_path)):
            print("path doesn't exist")
            continue
        src_root = ElementTree.parse(src_annot_path).getroot()
        src_objects = src_root.findall("object")

        dst_root = ElementTree.parse(dst_annot_path).getroot()
        dst_objects = dst_root.findall("object")

        for s_obj in src_objects:
            s_obj_id = s_obj.find("track_id").text

            for d_obj in dst_objects:
                d_obj_id = d_obj.find("track_id").text

                if s_obj_id == d_obj_id:
                    s_box = s_obj.find("bndbox")
                    s_xmin = s_box[0].text
                    s_ymin = s_box[1].text
                    s_xmax = s_box[2].text
                    s_ymax = s_box[3].text

                    d_box = d_obj.find("bndbox")
                    d_xmin = d_box[0].text
                    d_ymin = d_box[1].text
                    d_xmax = d_box[2].text
                    d_ymax = d_box[3].text

                    # add bottom mid points to src and dst points
                    s_width = int(s_xmax) - int(s_xmin)
                    s_mid_x = int(s_xmin) + s_width / 2
                    src_points.append([s_mid_x, s_ymax])
                    # src_points.append([s_xmin, s_ymax])
                    # src_points.append([s_xmax, s_ymax])

                    d_width = int(d_xmax) - int(d_xmin)
                    d_mid_x = int(d_xmin) + d_width / 2
                    dst_points.append([d_mid_x, d_ymax])
                    # dst_points.append([d_xmin, d_ymax])
                    # dst_points.append([d_xmax, d_ymax])
                    break

    # visualize points
    if DEBUG:
        src_img_name = "frame_{}_{:04d}.jpg".format(src_cam, i)
        src_img_path = "{}/{}".format(img_dir, src_img_name)
        src_img = cv2.imread(src_img_path)

        dst_img_name = "frame_{}_{:04d}.jpg".format(dst_cam, i)
        dst_img_path = "{}/{}".format(img_dir, dst_img_name)
        dst_img = cv2.imread(dst_img_path)

        # draw all trainig points
        for s_point, d_point in zip(src_points, dst_points):
            s_point = list(map(int, s_point))
            d_point = list(map(int, d_point))

            cv2.circle(src_img, (s_point[0], s_point[1]), 2, (0, 255, 0), 1)
            cv2.circle(dst_img, (d_point[0], d_point[1]), 2, (0, 255, 0), 1)

        cv2.imshow("src_img", src_img)
        cv2.imshow("dst_img", dst_img)
        cv2.waitKey(-1)

    # fit linear regresssion
    fit_linear_regression_bt_pts(src_points=src_points, dst_points=dst_points)

    # # create coordinate_mapping
    # src_points = np.array(src_points, dtype=np.float)
    # dst_points = np.array(dst_points, dtype=np.float)
    #
    # print("src_points : {}".format(src_points))
    # print("dst_points : {}".format(dst_points))
    #
    # coordinate_mapping, status = cv2.findHomography(srcPoints=src_points, dstPoints=dst_points, method=cv2.RANSAC)
    #
    # print("coordinate_mapping : {}\n".format(coordinate_mapping))
    #
    # with open("homography_87_bottom_pts", "wb") as output:
    #     pickle.dump(coordinate_mapping, output)
