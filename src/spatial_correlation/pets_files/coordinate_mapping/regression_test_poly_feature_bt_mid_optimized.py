"""
test the linear regression models created using polynomial features, of degree d. Metric used is IoU score
between estimated bounding box and actual bounding box in a camera view.
"""

import cv2
import os
import sys
import numpy as np
from xml.etree import ElementTree
import pickle
import pandas as pd
from matplotlib import pyplot as plt
import math
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge


def bb_iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return np.round(iou, decimals=2)


def plot_regression_results(iou_values):
    iou_ticks = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    iou_fractions = []
    for tick in iou_ticks:
        # find fraction of iou values above this iou
        target_iou = [j for j in iou_values if j >= tick]
        total_iou = len(iou_values)
        fraction = len(target_iou) / total_iou
        iou_fractions.append(np.round(fraction * 100.0, decimals=2))

    for k, t in enumerate(iou_fractions):
        print(iou_ticks[k], int(t))
    # print(iou_fractions)


def compute_iou_score(pred_coords, actual_coords):
    """
    :param pred_bbox: [bottom mid center (X,Y), width, height]
    :param actual_bbox:
    :return:
    """
    # compute top & bottom points of bboxes
    pred_bbox = []  # x1 y1 x2 y2
    pred_bbox.append(int(pred_coords[0] - pred_coords[2] / 2))
    pred_bbox.append(int(pred_coords[1] - pred_coords[3]))
    pred_bbox.append(int(pred_coords[0] + pred_coords[2] / 2))
    pred_bbox.append(int(pred_coords[1]))

    actual_bbox = []
    actual_bbox.append(int(actual_coords[0] - actual_coords[2] / 2))
    actual_bbox.append(int(actual_coords[1] - actual_coords[3]))
    actual_bbox.append(int(actual_coords[0] + actual_coords[2] / 2))
    actual_bbox.append(int(actual_coords[1]))

    score = bb_iou(boxA=pred_bbox, boxB=actual_bbox)
    return score


def load_scalar():
    scalar = None
    file_name = "regression_models/min_max_scalar_src_{}_dst_{}_PETS".format(src_cam, dst_cam)
    with open(file_name, 'rb') as ifile:
        scalar = pickle.load(ifile)  # load minmaxscalar
        assert scalar is not None
    return scalar


def load_regression_model():
    model = None
    model_file_path = "regression_models/lin_reg_deg_{}_src_{}_dst_{}_optimized".format(degree, src_cam, dst_cam)
    print("degree: {}, src_cam: {}\n".format(degree, src_cam))
    with open(model_file_path, 'rb') as input_file:
        model = pickle.load(input_file)
    return model


def test_poly_feature_linear_reg_bt_pt_height_width(s_points, d_points):
    """
    test polynomial feature linear regression model.
    Metric: IoU score with ground truth & predicted bounding boxes
    :param s_points:
    :param d_points:
    :return:
    """
    poly_features = PolynomialFeatures(degree=degree, interaction_only=False)
    min_max_scalar = None
    reg_model = None

    if len(s_points) != len(d_points):
        print("ERROR! Points not same.")
        return
    points = []
    for index in range(len(s_points)):
        points.append(s_points[index] + d_points[index])
    df = pd.DataFrame(data=points,
                      columns=['src_xmin', 'src_ymin', 'src_xmax', 'src_ymax', 'dst_xmin', 'dst_ymin', 'dst_xmax',
                               'dst_ymax'])
    df = df.astype('float')
    df['src_width'] = df['src_xmax'] - df['src_xmin']
    df['src_height'] = df['src_ymax'] - df['src_ymin']
    df['src_bt_mid_x'] = df['src_xmin'] + df['src_width'] / 2
    # df['src_bt_mid_y'] = df['src_ymax']
    df['dst_width'] = df['dst_xmax'] - df['dst_xmin']
    df['dst_height'] = df['dst_ymax'] - df['dst_ymin']
    df['dst_bt_mid_x'] = df['dst_xmin'] + df['dst_width'] / 2

    df = df.astype('int32')
    X = df[['src_bt_mid_x', 'src_ymax', 'src_width', 'src_height']]
    Y = df[['dst_bt_mid_x', 'dst_ymax', 'dst_width', 'dst_height']]
    # convert to 2d np array
    X = X.values
    Y = Y.values

    if NORMALIZE:
        min_max_scalar = load_scalar()
        X = min_max_scalar.transform(X)
        # Y = min_max_scalar.transform(Y)

    X_poly = poly_features.fit_transform(X)
    # #################### LOAD REGRESSION MODEL ######################################
    reg_model = load_regression_model()
    assert reg_model is not None
    # r_score = reg_model.score(X=X_poly, y=Y)
    # rmse = np.sqrt(mean_squared_error(y_true=Y, y_pred=reg_model.predict(X_poly)))
    # print("Test!! RMSE : {}, R2: {}".format(rmse, r_score))
    # sys.exit(-1)
    # ##################### COMPUTE IoU SCORES ########################################
    iou_scores = []
    for index, row in enumerate(X_poly):
        row = np.reshape(row, newshape=(1, -1))
        row_pred = reg_model.predict(row)
        row_pred = min_max_scalar.inverse_transform(row_pred)
        row_pred = np.array(row_pred, dtype=np.int32)
        # score = compute_iou_score(row_pred[0], actual_bbox=Y[index])
        score = compute_iou_score(pred_coords=row_pred[0], actual_coords=Y[index])
        iou_scores.append(score)
    plot_regression_results(iou_scores)
    plt.show()


if __name__ == "__main__":
    DEBUG = False
    NORMALIZE = True
    REGULARIZE = True
    DATASET_DIR = "../../../../dataset"
    img_dir = "{}/PETS_org/JPEGImages".format(DATASET_DIR)
    annot_dir = "{}/PETS_org/Annotations".format(DATASET_DIR)

    src_cam = 8  # collab cam
    dst_cam = 5  # ref cam
    degree = 4
    src_points = []
    dst_points = []

    # read training frame names
    test_frames = np.loadtxt("{}/PETS_1/ImageSets/test.txt".format(DATASET_DIR),
                             dtype=np.int32)
    for i in test_frames:  # last 30% frames kept for testing
        # for i in range(548, 796):  # last 30% frames kept for testing
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
                    src_points.append([s_xmin, s_ymin, s_xmax, s_ymax])

                    d_box = d_obj.find("bndbox")
                    d_xmin = d_box[0].text
                    d_ymin = d_box[1].text
                    d_xmax = d_box[2].text
                    d_ymax = d_box[3].text
                    dst_points.append([d_xmin, d_ymin, d_xmax, d_ymax])
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

            cv2.circle(src_img, (s_point[2], s_point[3]), 2, (0, 255, 0), 1)
            cv2.circle(dst_img, (d_point[2], d_point[3]), 2, (0, 255, 0), 1)

        cv2.imshow("src_img_{}".format(i), src_img)
        cv2.imshow("dst_img_{}".format(i), dst_img)
        cv2.waitKey(0)
    print("\ntotal points: src {}, dst {}\n".format(len(src_points), len(dst_points)))
    test_poly_feature_linear_reg_bt_pt_height_width(s_points=src_points, d_points=dst_points)
