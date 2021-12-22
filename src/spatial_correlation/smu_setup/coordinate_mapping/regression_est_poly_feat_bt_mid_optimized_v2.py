"""
regression estimator for coordinate mapping from one frame to another. Linear regression
use multiple points from each bounding box. These points are on the road.
Optimized version -- including feature normalization, regularization etc.
main file used for coordinate mapping and IoU vs Degree analysis as well
"""

import cv2
import os
import numpy as np
from xml.etree import ElementTree
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error
import json


def load_scalar():
    scalar = None
    file_name = "regression_models/min_max_scalar_src_{}_dst_{}_PETS".format(src_cam, dst_cam)
    with open(file_name, 'rb') as ifile:
        scalar = pickle.load(ifile)  # load minmaxscalar
        assert scalar is not None
    return scalar


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
    print(np.array(iou_fractions))
    # print(iou_fractions)


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


def create_df(s_points, d_points):
    points = []
    for index in range(len(s_points)):
        points.append(s_points[index] + d_points[index])  # horizontally append both lists
    df = pd.DataFrame(data=points,
                      columns=['src_xmin', 'src_ymin', 'src_width', 'src_height', 'dst_xmin', 'dst_ymin', 'dst_width',
                               'dst_height'])
    df = df.astype('float')
    # df['src_width'] = df['src_xmax'] - df['src_xmin']
    # df['src_height'] = df['src_ymax'] - df['src_ymin']
    df['src_bt_mid_x'] = df['src_xmin'] + df['src_width'] / 2
    # df['src_bt_mid_y'] = df['src_ymax']
    # df['dst_width'] = df['dst_xmax'] - df['dst_xmin']
    # df['dst_height'] = df['dst_ymax'] - df['dst_ymin']
    df['dst_bt_mid_x'] = df['dst_xmin'] + df['dst_width'] / 2
    df['src_ymax'] = df['src_ymin'] + df['src_height']
    df['dst_ymax'] = df['dst_ymin'] + df['dst_height']
    return df


def load_regression_model():
    model = None
    model_file_path = "regression_models/lin_reg_deg_{}_src_{}_dst_{}_optimized".format(degree, src_cam, dst_cam)
    # print("degree: {}, src_cam: {}\n".format(degree, src_cam))
    with open(model_file_path, 'rb') as input_file:
        model = pickle.load(input_file)
    return model


def test_poly_feature_linear_reg_bt_pt_height_width(test_df):
    """
        test a linear regression, with polynomial features of a degree, to the data. Features used are
        bottom-mid point, height & width of bounding boxes.
        :param s_points:
        :param d_points:
        :return:
        """
    poly_features = PolynomialFeatures(degree=degree, interaction_only=False)
    # model = None
    # min_max_scalar = None

    # if len(s_points) != len(d_points):
    #     print("ERROR! Points not same.")
    #     return
    # # prepare data
    # points = []
    # for index in range(len(s_points)):
    #     points.append(s_points[index] + d_points[index])  # horizontally append both lists
    # df = pd.DataFrame(data=points,
    #                   columns=['src_xmin', 'src_ymin', 'src_xmax', 'src_ymax', 'dst_xmin', 'dst_ymin', 'dst_xmax',
    #                            'dst_ymax'])
    # df = df.astype('float')
    # df['src_width'] = df['src_xmax'] - df['src_xmin']
    # df['src_height'] = df['src_ymax'] - df['src_ymin']
    # df['src_bt_mid_x'] = df['src_xmin'] + df['src_width'] / 2
    # # df['src_bt_mid_y'] = df['src_ymax']
    # df['dst_width'] = df['dst_xmax'] - df['dst_xmin']
    # df['dst_height'] = df['dst_ymax'] - df['dst_ymin']
    # df['dst_bt_mid_x'] = df['dst_xmin'] + df['dst_width'] / 2
    #
    # # df = df.astype('int32')
    X = test_df[['src_bt_mid_x', 'src_ymax', 'src_width', 'src_height']]
    Y = test_df[['dst_bt_mid_x', 'dst_ymax', 'dst_width', 'dst_height']]
    # convert to 2d np array
    X = X.values
    Y = Y.values

    # NORMALIZATION
    if NORMALIZE:
        min_max_scalar = load_scalar()
        X = min_max_scalar.transform(X)
        Y = min_max_scalar.transform(Y)
    X_poly = poly_features.fit_transform(X)
    # #################### LOAD REGRESSION MODEL ######################################
    reg_model = load_regression_model()
    assert reg_model is not None
    r_score = reg_model.score(X=X_poly, y=Y)
    rmse = np.sqrt(mean_squared_error(y_true=Y, y_pred=reg_model.predict(X_poly)))
    print("Test!! RMSE : {}, R2: {}".format(round(rmse), np.round(r_score, decimals=2)))

    # ##################### COMPUTE IoU SCORES ########################################
    if not IOU_ANALYSIS:
        return
    iou_scores = []
    for index, row in enumerate(X_poly):
        row = np.reshape(row, newshape=(1, -1))
        row_pred = reg_model.predict(row)
        if NORMALIZE:
            row_pred = min_max_scalar.inverse_transform(row_pred)
        row_pred = np.array(row_pred, dtype=np.int32)
        # score = compute_iou_score(row_pred[0], actual_bbox=Y[index])
        score = compute_iou_score(pred_coords=row_pred[0], actual_coords=Y[index])
        iou_scores.append(score)
    plot_regression_results(iou_scores)


def fit_poly_feature_linear_reg_bt_pt_height_width(train_df):
    """
    fits a linear regression, with polynomial features of a degree, to the data. Features used are
    bottom-mid point, height & width of bounding boxes.
    :param s_points:
    :param d_points:
    :return:
    """
    poly_features = PolynomialFeatures(degree=degree, interaction_only=False)
    model = LinearRegression(normalize=True)
    # model = Ridge(alpha=100)
    # model = SVR()
    # model = RandomForestRegressor(max_features=5)
    # model = GradientBoostingRegressor()
    # min_max_scalar = StandardScaler()

    # visualize points
    if DEBUG:
        src_img_name = "frame_{}_{}.jpg".format(src_cam, 0)
        src_img_path = "{}/{}".format(img_dir.format(src_cam), src_img_name)
        src_img = cv2.imread(src_img_path)

        dst_img_name = "frame_{}_{}.jpg".format(src_cam, 0)
        dst_img_path = "{}/{}".format(img_dir.format(dst_cam), dst_img_name)
        dst_img = cv2.imread(dst_img_path)

        # draw all trainig points
        s_points = train_df[['src_bt_mid_x', 'src_ymax']]
        s_points["src_bt_mid_x"] = s_points["src_bt_mid_x"].astype(int)
        d_points = train_df[['dst_bt_mid_x', 'dst_ymax']]
        d_points["dst_bt_mid_x"] = d_points["dst_bt_mid_x"].astype(int)

        for index, s_point in s_points.iterrows():
            # s_point = list(map(int, s_point))
            # d_point = list(map(int, d_point))
            d_point = d_points.iloc[index]
            cv2.circle(src_img, (s_point[0], s_point[1]), 2, (0, 255, 0), 1)
            cv2.circle(dst_img, (d_point[0], d_point[1]), 2, (0, 255, 0), 1)

        cv2.imshow("src_img", src_img)
        cv2.imshow("dst_img", dst_img)
        cv2.waitKey(100)

    # df = df.astype('int32')
    X = train_df[['src_bt_mid_x', 'src_ymax', 'src_width', 'src_height']]
    Y = train_df[['dst_bt_mid_x', 'dst_ymax', 'dst_width', 'dst_height']]
    # convert to 2d np array
    X = X.values
    Y = Y.values

    # NORMALIZATION
    if NORMALIZE:
        min_max_scalar = min_max_scalar.fit(X)
        X = min_max_scalar.transform(X)
        Y = min_max_scalar.transform(Y)
    X_poly = poly_features.fit_transform(X)
    # ################### cross-validation ##########################
    # kfolds = KFold(n_splits=5, shuffle=True, random_state=10)
    # r2_list = []
    # rmse_list = []
    # r2_training_list = []
    # best_model = None
    # min_rmse = 100
    # for train, test in kfolds.split(X_poly):
    #     X_poly_train = X_poly[train]
    #     X_poly_test = X_poly[test]
    #     Y_train = Y[train]
    #     Y_test = Y[test]
    #
    #     # print("\ntrain points: {}, test points: {}\n".format(len(X_poly_train), len(X_poly_test)))
    #     model.fit(X=X_poly_train, y=Y_train)
    #     # training accuracy
    #     X_poly_test_pred = model.predict(X_poly_test)
    #     # r2_train = model.score(X=Y_test, y=)
    #     r2 = r2_score(y_true=Y_test, y_pred=X_poly_test_pred)
    #     r2 = np.round(r2, decimals=2)
    #     rmse = np.sqrt(mean_squared_error(y_true=Y_test, y_pred=X_poly_test_pred))
    #     rmse = np.round(rmse, decimals=2)
    #     # print("r2 : {}, rmse :{}\n".format(r2, rmse))
    #     r2_list.append(r2)
    #     # r2_training_list.append(r2_train)
    #     rmse_list.append(rmse)
    #     if rmse < min_rmse:
    #         best_model = model
    #         min_rmse = rmse
    # print("\nR2_Raw: {}, R2_mean: {:.2f}, R2_Std: {:.2f}".format(r2_list, np.mean(r2_list), np.std(r2_list)))
    # print("\nRMSE_Raw: {}, RMSE_mean: {}, RMSE_Std: {}".format(rmse_list, np.mean(rmse_list), np.std(rmse_list)))
    # print("R2_train: {}".format(r2_training_list))

    # write model to a file
    model.fit(X=X_poly, y=Y)
    # training set accuracy
    r_score = model.score(X=X_poly, y=Y)
    rmse = np.sqrt(mean_squared_error(y_true=Y, y_pred=model.predict(X_poly)))
    print("Training!! RMSE : {}, R2: {}".format(round(rmse), np.round(r_score, decimals=2)))

    # with open("./regression_models/min_max_scalar_src_{}_dst_{}_CAMPUS".format(src_cam, dst_cam), 'wb') as ofile:
    #     pickle.dump(min_max_scalar, ofile)
    with open("./regression_models/lin_reg_deg_{}_src_{}_dst_{}_optimized".format(degree, src_cam, dst_cam),
              "wb") as ofile:
        pickle.dump(model, ofile)


def create_filenames(cam, f_numbers_file):
    filenames = []
    with open(f_numbers_file) as input_file:
        names = input_file.read().split("\n")
        for n in names:
            if len(n) == 0:
                continue
            f_name = "frame_{}_{}.json".format(cam, n.strip())
            filenames.append(f_name)
    return filenames


if __name__ == "__main__":
    DEBUG = False
    NORMALIZE = False
    REGULARIZE = False
    img_dir = r"../../../../rpi_hardware/raw_image_processing/data/episode_1/pi_{}_frames_1056_v2/"
    annot_dir = r"../../../../rpi_hardware/raw_image_processing/data/episode_1/ground_truth/frame_wise_gt_json/"
    train_f_numbers = r"./train_frame_numbers.txt"
    test_f_numbers = r"./test_frame_numbers.txt"
    IOU_ANALYSIS = True

    src_cam = 1  # collab cam
    dst_cam = 2  # ref cam
    degree = 4

    s_pts_train = []  # source/dst points for testing and training
    d_pts_train = []

    print("Degree: {}, Normalize: {}, Regularize: {}, src_cam : {}, dst_cam: {}".format(degree, NORMALIZE, REGULARIZE,
                                                                                        src_cam, dst_cam))
    # read training frame names
    # trainval_frames = np.loadtxt("{}/PETS_org/ImageSets/Main/coord_mapping_trainval_70.txt".format(DATASET_DIR),
    #                              dtype=np.int32)

    src_dir = "{}/cam_{}".format(annot_dir, src_cam)
    dst_dir = "{}/cam_{}".format(annot_dir, dst_cam)

    # src_filenames = os.listdir(src_dir)
    train_filenames = create_filenames(src_cam, train_f_numbers)
    # src_filenames = sorted(src_filenames, key=lambda x: int(x[:-5].split("_")[2]))
    for name in train_filenames:
        # print(name)
        f_num = name[:-5].split("_")[2]
        dst_annot_name = "frame_{}_{}.json".format(dst_cam, f_num)

        with open("{}/{}".format(src_dir, name)) as src_file:
            src_json = json.load(src_file)
            with open("{}/{}".format(dst_dir, dst_annot_name)) as dst_file:
                dst_json = json.load(dst_file)
                if len(src_json) == 0 or len(dst_json) == 0:
                    continue
                # extract common objects
                for src_obj in src_json:
                    src_obj_id = src_obj["obj_id"]
                    for dst_obj in dst_json:
                        if src_obj_id == dst_obj["obj_id"]:
                            # clip values to a range
                            src_obj_coords = src_obj["coords"]
                            dst_obj_coords = dst_obj["coords"]
                            src_obj_coords = np.clip(src_obj_coords, a_min=0, a_max=1056)
                            dst_obj_coords = np.clip(dst_obj_coords, a_min=0, a_max=1056)
                            # add objects to the list
                            # src_points = [src_obj_coords[0], src_obj_coords[1], src_obj_coords[0] + src_obj_coords[2],
                            #               src_obj_coords[1] + src_obj_coords[3]]
                            # src_points = np.clip(src_points, a_min=0, a_max=1056)
                            # dst_points = [dst_obj_coords[0], dst_obj_coords[1], dst_obj_coords[0] + dst_obj_coords[2],
                            #               dst_obj_coords[1] + dst_obj_coords[3]]
                            # dst_points = np.clip(dst_points, a_min=0, a_max=1056)
                            s_pts_train.append(src_obj_coords.tolist())
                            d_pts_train.append(dst_obj_coords.tolist())
                            continue

    # for i in range(0, 2000, 5):
    #     src_annot_name = "C{}_{:08d}.xml".format(src_cam, i)
    #     src_annot_path = "{}/{}".format(annot_dir, src_annot_name)
    #
    #     dst_annot_name = "C{}_{:08d}.xml".format(dst_cam, i)
    #     dst_annot_path = "{}/{}".format(annot_dir, dst_annot_name)
    #
    #     if not (os.path.exists(src_annot_path) and os.path.exists(dst_annot_path)):
    #         print("path doesn't exist")
    #         continue
    #     src_root = ElementTree.parse(src_annot_path).getroot()
    #     src_objects = src_root.findall("object")
    #
    #     dst_root = ElementTree.parse(dst_annot_path).getroot()
    #     dst_objects = dst_root.findall("object")
    #
    #     for s_obj in src_objects:
    #         s_obj_id = s_obj.find("track_id").text
    #         for d_obj in dst_objects:
    #             d_obj_id = d_obj.find("track_id").text
    #             if s_obj_id == d_obj_id:
    #                 s_box = s_obj.find("bndbox")
    #                 s_xmin = s_box[0].text
    #                 s_ymin = s_box[1].text
    #                 s_xmax = s_box[2].text
    #                 s_ymax = s_box[3].text
    #                 if i <= TRAINING_FRAMES:
    #                     s_pts_train.append([s_xmin, s_ymin, s_xmax, s_ymax])
    #                 else:
    #                     s_pts_test.append([s_xmin, s_ymin, s_xmax, s_ymax])
    #
    #                 d_box = d_obj.find("bndbox")
    #                 d_xmin = d_box[0].text
    #                 d_ymin = d_box[1].text
    #                 d_xmax = d_box[2].text
    #                 d_ymax = d_box[3].text
    #                 if i <= TRAINING_FRAMES:
    #                     d_pts_train.append([d_xmin, d_ymin, d_xmax, d_ymax])
    #                 else:
    #                     d_pts_test.append([d_xmin, d_ymin, d_xmax, d_ymax])
    #                 break

    # separate training and test frames -- 70:30 split
    train = create_df(s_pts_train, d_pts_train)
    # train, test = train_test_split(dataframe, test_size=0.30, random_state=20, shuffle=True)

    print("\nTraining points: src {}\n".format(len(train)))
    fit_poly_feature_linear_reg_bt_pt_height_width(train)

    ################### TEST PERFORMANCE ####################################################
    test_filenames = create_filenames(src_cam, test_f_numbers)
    # src_filenames = sorted(src_filenames, key=lambda x: int(x[:-5].split("_")[2]))
    s_pts_train = []  # source/dst points for testing and training
    d_pts_train = []
    for name in test_filenames:
        # print(name)
        f_num = name[:-5].split("_")[2]
        dst_annot_name = "frame_{}_{}.json".format(dst_cam, f_num)

        with open("{}/{}".format(src_dir, name)) as src_file:
            src_json = json.load(src_file)
            with open("{}/{}".format(dst_dir, dst_annot_name)) as dst_file:
                dst_json = json.load(dst_file)
                if len(src_json) == 0 or len(dst_json) == 0:
                    continue
                # extract common objects
                for src_obj in src_json:
                    src_obj_id = src_obj["obj_id"]
                    for dst_obj in dst_json:
                        if src_obj_id == dst_obj["obj_id"]:
                            # clip values to a range
                            src_obj_coords = src_obj["coords"]
                            dst_obj_coords = dst_obj["coords"]
                            src_obj_coords = np.clip(src_obj_coords, a_min=0, a_max=1056)
                            dst_obj_coords = np.clip(dst_obj_coords, a_min=0, a_max=1056)
                            s_pts_train.append(src_obj_coords.tolist())
                            d_pts_train.append(dst_obj_coords.tolist())
                            continue
    test = create_df(s_pts_train, d_pts_train)
    print("\nTest points: src {}\n".format(len(test)))
    test_poly_feature_linear_reg_bt_pt_height_width(test)
