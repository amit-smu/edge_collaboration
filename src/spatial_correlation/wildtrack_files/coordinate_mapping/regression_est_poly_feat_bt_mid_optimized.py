"""
regression estimator for coordinate mapping from one frame to another. Linear regression
use multiple points from each bounding box. These points are on the road.
Optimized version -- including feature normalization, regularization etc.
main file used for coordinate mapping
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
    # print("degree: {}, src_cam: {}\n".format(degree, src_cam))
    with open(model_file_path, 'rb') as input_file:
        model = pickle.load(input_file)
    return model


def test_poly_feature_linear_reg_bt_pt_height_width(s_points, d_points):
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

    if len(s_points) != len(d_points):
        print("ERROR! Points not same.")
        return
    # prepare data
    points = []
    for index in range(len(s_points)):
        points.append(s_points[index] + d_points[index])  # horizontally append both lists
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

    # df = df.astype('int32')
    X = df[['src_bt_mid_x', 'src_ymax', 'src_width', 'src_height']]
    Y = df[['dst_bt_mid_x', 'dst_ymax', 'dst_width', 'dst_height']]
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


def fit_poly_feature_linear_reg_bt_pt_height_width(s_points, d_points):
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
    # model = RandomForestRegressor(max_features=2)
    # model = GradientBoostingRegressor()
    min_max_scalar = StandardScaler()

    if len(s_points) != len(d_points):
        print("ERROR! Points not same.")
        return
    # prepare data
    points = []
    for index in range(len(s_points)):
        points.append(s_points[index] + d_points[index])  # horizontally append both lists
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

    # visualize points
    if DEBUG:
        src_img_name = "frame_{}_{:04d}.jpg".format(src_cam, 15)
        src_img_path = "{}/{}".format(img_dir, src_img_name)
        src_img = cv2.imread(src_img_path)

        dst_img_name = "frame_{}_{:04d}.jpg".format(dst_cam, 15)
        dst_img_path = "{}/{}".format(img_dir, dst_img_name)
        dst_img = cv2.imread(dst_img_path)

        # draw all trainig points
        s_points = df[['src_bt_mid_x', 'src_ymax']]
        s_points["src_bt_mid_x"] = s_points["src_bt_mid_x"].astype(int)
        d_points = df[['dst_bt_mid_x', 'dst_ymax']]
        d_points["dst_bt_mid_x"] = d_points["dst_bt_mid_x"].astype(int)

        for index, s_point in s_points.iterrows():
            # s_point = list(map(int, s_point))
            # d_point = list(map(int, d_point))
            d_point = d_points.iloc[index]
            cv2.circle(src_img, (s_point[0], s_point[1]), 2, (0, 255, 0), 1)
            cv2.circle(dst_img, (d_point[0], d_point[1]), 2, (0, 255, 0), 1)

        cv2.imshow("src_img_{}".format(i), src_img)
        cv2.imshow("dst_img_{}".format(i), dst_img)
        cv2.waitKey(0)

    # df = df.astype('int32')
    X = df[['src_bt_mid_x', 'src_ymax', 'src_width', 'src_height']]
    Y = df[['dst_bt_mid_x', 'dst_ymax', 'dst_width', 'dst_height']]
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

    with open("./regression_models/min_max_scalar_src_{}_dst_{}_WT".format(src_cam, dst_cam), 'wb') as ofile:
        pickle.dump(min_max_scalar, ofile)
    with open("./regression_models/lin_reg_deg_{}_src_{}_dst_{}_optimized".format(degree, src_cam, dst_cam),
              "wb") as ofile:
        pickle.dump(model, ofile)


if __name__ == "__main__":
    DEBUG = False
    NORMALIZE = False
    REGULARIZE = False
    img_dir = r"G:\Datasets\Wildtrack_dataset\PNGImages"
    annot_dir = r"G:\Datasets\Wildtrack_dataset\Annotations"

    src_cam = 3  # collab cam
    dst_cam = 5  # ref cam
    degree = 1
    TRAINING_FRAMES = 1400  # out of 400 total, 119 frames kept for testing. Frame no range = (0, 2000)

    s_pts_train = []  # source/dst points for testing and training
    d_pts_train = []
    s_pts_test = []
    d_pts_test = []

    print("Degree: {}, Normalize: {}, Regularize: {}".format(degree, NORMALIZE, REGULARIZE))
    # read training frame names
    # trainval_frames = np.loadtxt("{}/PETS_org/ImageSets/Main/coord_mapping_trainval_70.txt".format(DATASET_DIR),
    #                              dtype=np.int32)
    for i in range(0, 2000, 5):
        src_annot_name = "C{}_{:08d}.xml".format(src_cam, i)
        src_annot_path = "{}/{}".format(annot_dir, src_annot_name)

        dst_annot_name = "C{}_{:08d}.xml".format(dst_cam, i)
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
                    if i <= TRAINING_FRAMES:
                        s_pts_train.append([s_xmin, s_ymin, s_xmax, s_ymax])
                    else:
                        s_pts_test.append([s_xmin, s_ymin, s_xmax, s_ymax])

                    d_box = d_obj.find("bndbox")
                    d_xmin = d_box[0].text
                    d_ymin = d_box[1].text
                    d_xmax = d_box[2].text
                    d_ymax = d_box[3].text
                    if i <= TRAINING_FRAMES:
                        d_pts_train.append([d_xmin, d_ymin, d_xmax, d_ymax])
                    else:
                        d_pts_test.append([d_xmin, d_ymin, d_xmax, d_ymax])
                    break

    print("\nTraining points: src {}, dst {}\n".format(len(s_pts_train), len(d_pts_train)))
    fit_poly_feature_linear_reg_bt_pt_height_width(s_points=s_pts_train, d_points=d_pts_train)

    print("\nTest points: src {}, dst {}\n".format(len(s_pts_test), len(d_pts_test)))
    test_poly_feature_linear_reg_bt_pt_height_width(s_points=s_pts_test, d_points=d_pts_test)
