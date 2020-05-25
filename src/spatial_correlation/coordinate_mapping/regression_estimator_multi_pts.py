"""
regression estimator for coordinate mapping from one frame to another. Linear regression
use multiple points from each bounding box. These points are on the road.
Final file used for regression
"""

import cv2
import os
import numpy as np
from xml.etree import ElementTree
import pickle
import pandas as pd
from matplotlib import pyplot as plt
import math
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LassoCV, MultiTaskLassoCV
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score, mean_squared_error


def fit_lin_reg_bt_pts_height(s_points, d_points):
    """use bottom points and height
        :param s_points:
        :param d_points:
        :return:
        """
    regressor = LinearRegression()

    # add data to df
    points = []
    for index in range(len(s_points)):
        points.append(s_points[index] + d_points[index])

    df = pd.DataFrame(data=points,
                      columns=['src_xmin', 'src_ymin', 'src_xmax', 'src_ymax', 'dst_xmin', 'dst_ymin', 'dst_xmax',
                               'dst_ymax'])
    df = df.astype('int32')

    # create bottom points
    df['src_height'] = df['src_ymax'] - df['src_ymin']
    df['dst_height'] = df['dst_ymax'] - df['dst_ymin']

    X = df[['src_xmin', 'src_ymax', 'src_xmax', 'src_ymax', 'src_height']].values
    Y = df[['dst_xmin', 'dst_ymax', 'dst_xmax', 'dst_ymax', 'dst_height']].values

    regressor.fit(X=X, y=Y)
    print(regressor.coef_)
    print(regressor.intercept_)

    # save the model to file
    with open("regression_{}_bt_pts_height".format(src_cam), "wb") as output:
        pickle.dump(regressor, output)
        print("Saved model : regression_{}_bt_pts_height\n".format(src_cam))


def fit_linear_reg_bt_pt_diagonal(s_points, d_points):
    """
    use bottom mid point and diagonal
    :param s_points:
    :param d_points:
    :return:
    """
    regressor = LinearRegression()

    # add data to df
    points = []
    for index in range(len(s_points)):
        points.append(s_points[index] + d_points[index])

    df = pd.DataFrame(data=points,
                      columns=['src_xmin', 'src_ymin', 'src_xmax', 'src_ymax', 'dst_xmin', 'dst_ymin', 'dst_xmax',
                               'dst_ymax'])
    df = df.astype('int32')

    # compute bottom mid point
    df['src_bt_mid_x'] = (df['src_xmin'] + df['src_xmax']) / 2
    df['src_bt_mid_y'] = (df['src_ymin'] + df['src_ymax']) / 2
    df['dst_bt_mid_x'] = (df['dst_xmin'] + df['dst_xmax']) / 2
    df['dst_bt_mid_y'] = (df['dst_ymin'] + df['dst_ymax']) / 2

    # compute diagonal values
    df['src_diagonal'] = ((df['src_xmin'] - df['src_xmax']) ** 2 + (df['src_ymin'] - df['src_ymax']) ** 2) ** 0.5
    df['dst_diagonal'] = ((df['dst_xmin'] - df['dst_xmax']) ** 2 + (df['dst_ymin'] - df['dst_ymax']) ** 2) ** 0.5

    X = df[['src_bt_mid_x', 'src_bt_mid_y', 'src_diagonal']].values
    Y = df[['dst_bt_mid_x', 'dst_bt_mid_y', 'dst_diagonal']].values
    regressor.fit(X=X, y=Y)
    print(regressor.coef_)

    # save the model to file
    with open("regression_{}_bt_diag_".format(src_cam), "wb") as output:
        pickle.dump(regressor, output)
        print("Saved model : regression_{}_bt_diag\n".format(src_cam))


def fit_linear_reg_bt_pt_height_width(s_points, d_points):
    regressor = LinearRegression()

    # add data to df
    points = []
    for index in range(len(s_points)):
        points.append(s_points[index] + d_points[index])

    df = pd.DataFrame(data=points,
                      columns=['src_xmin', 'src_ymin', 'src_xmax', 'src_ymax', 'dst_xmin', 'dst_ymin', 'dst_xmax',
                               'dst_ymax'])
    # df = pd.DataFrame(data=points,
    #                   columns=['s_xmin', 's_ymin', 's_xmax', 's_ymax', 'd_xmin', 'd_ymin', 'd_xmax', 'd_ymax', ])
    df = df.astype('int32')

    # X = df[['s_xmin', 's_ymax', 's_xmax', 's_ymax']].values
    # Y = df[['d_xmin', 'd_ymax', 'd_xmax', 'd_ymax']].values
    X = df[['src_bot_x', 'src_bot_y', 'src_width', 'src_height']].values
    Y = df[['dst_bot_x', 'dst_bot_y', 'dst_width', 'dst_height']].values

    # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=100)
    regressor.fit(X=X, y=Y)
    print(regressor.coef_)

    # save the model to file
    with open("regression_{}_multi_pts".format(src_cam), "wb") as output:
        pickle.dump(regressor, output)
        print("Saved model : regression_{}_multi_pts\n".format(src_cam))


def visualize_mid_points(s_points, d_points):
    # add data to df
    points = []
    for index in range(len(s_points)):
        points.append(s_points[index] + d_points[index])

    df = pd.DataFrame(data=points,
                      columns=['src_xmin', 'src_ymin', 'src_xmax', 'src_ymax', 'dst_xmin', 'dst_ymin', 'dst_xmax',
                               'dst_ymax'])
    # df = pd.DataFrame(data=points,
    #                   columns=['s_xmin', 's_ymin', 's_xmax', 's_ymax', 'd_xmin', 'd_ymin', 'd_xmax', 'd_ymax', ])

    df = df.astype('int32')
    df['src_bt_mid_x'] = (df['src_xmin'] + df['src_xmax']) / 2
    df['src_bt_mid_y'] = (df['src_ymin'] + df['src_ymax']) / 2

    df['dst_bt_mid_x'] = (df['dst_xmin'] + df['dst_xmax']) / 2
    df['dst_bt_mid_y'] = (df['dst_ymin'] + df['dst_ymax']) / 2

    df = df.astype('int32')

    # X = df[['s_xmin', 's_ymax', 's_xmax', 's_ymax']].values
    # Y = df[['d_xmin', 'd_ymax', 'd_xmax', 'd_ymax']].values
    X = df['src_bt_mid_y'].values
    Y = df['dst_bt_mid_y'].values
    # plt.scatter(x=X, y=Y)
    plt.subplot(121)
    plt.plot(X, Y)
    # plt.subplot(122)
    # plt.plot()
    plt.show()


def fit_fundamental_mat(s_points, d_points):
    print()
    s_pts = []
    d_pts = []

    for sp, dp in zip(s_points, d_points):
        s_pts.append((sp[0], sp[1]))
        s_pts.append((sp[2], sp[3]))
        d_pts.append((dp[0], dp[1]))
        d_pts.append((dp[2], dp[3]))

    s_pts = np.int32(s_pts)
    d_pts = np.int32(d_pts)
    f_mat, mask = cv2.findFundamentalMat(points1=s_pts, points2=d_pts, method=cv2.FM_RANSAC)
    print(f_mat)

    # test the matrix
    test_point = s_pts[0]
    test_point = np.reshape(test_point, (1, 2))
    # test_point = np.int32(test_point)
    line = cv2.computeCorrespondEpilines(test_point, 1, f_mat)
    print(line)


def split_train_test_data(input_data, training_fraction=1.0):
    """
    :param input_data: ndarray
    :return:
    """
    if training_fraction == 1.0:
        "use all data for training and same data for testing"
        train_data = input_data
        test_data = input_data
    else:
        total_rows = len(input_data)
        training_rows = int(training_fraction * total_rows)
        train_data = input_data[0:training_rows, :]
        test_data = input_data[training_rows:, :]
    return train_data, test_data


def random_frame_sampling():
    """
    randomly pick 681 frames for training (+validation) and 114 for testing the regression models
    :return:
    """
    out_dir = "../../../dataset/PETS_1/ImageSets"
    output_file_trainval_frames = "{}/trainval.txt".format(out_dir)
    output_file_testing_frames = "{}/test.txt".format(out_dir)
    frames_numbers = np.arange(0, 795, dtype=np.int)
    # with open(output_file_trainval_frames, 'w') as output_train, open(output_file_testing_frames, 'w') as output_test:
    kfold = KFold(n_splits=7, shuffle=True, random_state=10)
    for train, test in kfold.split(frames_numbers):
        # write test and train to files
        np.savetxt(fname=output_file_trainval_frames, X=train, fmt="%d")
        np.savetxt(fname=output_file_testing_frames, X=test, fmt="%d")
        print("training frames: {}, testing frames: {}".format(len(train), len(test)))
        break


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

    df = pd.DataFrame(data=points,
                      columns=['src_bot_x', 'src_bot_y', 'src_width', 'src_height', 'dst_bot_x', 'dst_bot_y',
                               'dst_width', 'dst_height'])
    # df = pd.DataFrame(data=points,
    #                   columns=['s_xmin', 's_ymin', 's_xmax', 's_ymax', 'd_xmin', 'd_ymin', 'd_xmax', 'd_ymax', ])
    df = df.astype('int32')

    # X = df[['s_xmin', 's_ymax', 's_xmax', 's_ymax']].values
    # Y = df[['d_xmin', 'd_ymax', 'd_xmax', 'd_ymax']].values
    X = df[['src_bot_x', 'src_bot_y', 'src_width', 'src_height']].values
    Y = df[['dst_bot_x', 'dst_bot_y', 'dst_width', 'dst_height']].values

    # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=100)
    regressor.fit(X=X, y=Y)
    print(regressor.coef_)

    # save the model to file
    with open("regression_{}_multi_pts".format(src_cam), "wb") as output:
        pickle.dump(regressor, output)
        print("Saved model : regression_{}_multi_pts\n".format(src_cam))
    print(regressor.predict([[120, 150]]))

    regressor = None
    with open("regression_{}_multi_pts".format(src_cam), "rb") as input_file:
        regressor = pickle.load(input_file)
    print(regressor.predict([[120, 150]]))
    print()


def fit_poly_feature_linear_reg_bt_pt_height_width(s_points, d_points):
    # Alpha (regularization strength) of LASSO regression
    # lasso_eps = 0.0001
    # lasso_nalpha = 20
    # lasso_iter = 7000
    # # Min and max degree of polynomials features to consider
    # degree_min = 2
    # degree_max = 4

    # prepare data
    points = []
    for index in range(len(s_points)):
        points.append(s_points[index] + d_points[index])
    df = pd.DataFrame(data=points,
                      columns=['src_xmin', 'src_ymin', 'src_xmax', 'src_ymax', 'dst_xmin', 'dst_ymin', 'dst_xmax',
                               'dst_ymax'])
    df = df.astype('int32')

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

    df = df.astype('int32')
    X = df[['src_bt_mid_x', 'src_ymax', 'src_width', 'src_height']]
    Y = df[['dst_bt_mid_x', 'dst_ymax', 'dst_width', 'dst_height']]
    # convert to 2d np array
    X = X.values
    Y = Y.values
    # Test/train split
    # X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30)
    # Make a pipeline model with polynomial transformation and LASSO regression with cross-validation, run it for increasing degree of polynomial (complexity of the model)
    # test_scores = []
    # RMSE_scores = []

    # for degree in range(degree_min, degree_max + 1):
    #     print(degree)
    #     model = make_pipeline(PolynomialFeatures(degree, interaction_only=False),
    #                           MultiTaskLassoCV(eps=lasso_eps, n_alphas=lasso_nalpha, max_iter=lasso_iter,
    #                                            normalize=True, cv=5))
    #     model.fit(X_train, y_train)
    #     test_pred = np.array(model.predict(X_test))
    #     RMSE = np.sqrt(np.sum(np.square(test_pred - y_test)))
    #     test_score = model.score(X_test, y_test)
    #     RMSE_scores.append(RMSE)
    #     test_scores.append(test_score)
    #     # break
    # x_axis = list(range(degree_min, degree_max + 1))
    # # x_axis = [2]
    # plt.plot(x_axis, test_scores, 'r')
    # plt.plot(x_axis, RMSE_scores, 'b')
    # plt.show()
    degree = 1
    poly_features = PolynomialFeatures(degree=degree, interaction_only=False)
    model = LinearRegression()
    # transform data

    X_poly = poly_features.fit_transform(X)

    #################### cross-validation ##########################
    kfolds = KFold(n_splits=5, shuffle=True, random_state=10)
    r2_list = []
    rmse_list = []
    best_model = None
    min_rmse = 100
    for train, test in kfolds.split(X_poly):
        X_poly_train = X_poly[train]
        X_poly_test = X_poly[test]
        Y_train = Y[train]
        Y_test = Y[test]

        # print("\ntrain points: {}, test points: {}\n".format(len(X_poly_train), len(X_poly_test)))
        model.fit(X=X_poly_train, y=Y_train)
        y_pred = model.predict(X_poly_test)
        r2 = r2_score(y_true=Y_test, y_pred=y_pred)
        r2 = np.round(r2, decimals=2)
        rmse = np.sqrt(mean_squared_error(y_true=Y_test, y_pred=y_pred))
        rmse = np.round(rmse, decimals=2)
        # print("r2 : {}, rmse :{}\n".format(r2, rmse))
        r2_list.append(r2)
        rmse_list.append(rmse)
        if rmse < min_rmse:
            best_model = model
            min_rmse = rmse
    print("\nR2_Raw: {}, R2_mean: {:.2f}, R2_Std: {:.2f}".format(r2_list, np.mean(r2_list), np.std(r2_list)))
    print("\nRMSE_Raw: {}, RMSE_mean: {}, RMSE_Std: {}".format(rmse_list, np.mean(rmse_list), np.std(rmse_list)))

    # write model to a file
    model_file_path = "regression_models/poly_feature_linear_regression_deg_{}_interaction_false_cam_{}".format(degree,
                                                                                                                src_cam)
    with open(model_file_path, 'wb') as output:
        pickle.dump(best_model, output)
        print("best_rmse : {}".format(min_rmse))

    ############################# TEMP ###################################
    # model = None
    # with open(model_file_path, 'rb') as input_file:
    #     model = pickle.load(input_file)
    # r2_list = []
    # rmse_list = []
    # for train, test in kfolds.split(X_poly):
    #     X_poly_train = X_poly[train]
    #     X_poly_test = X_poly[test]
    #     Y_train = Y[train]
    #     Y_test = Y[test]
    #
    #     # print("\ntrain points: {}, test points: {}\n".format(len(X_poly_train), len(X_poly_test)))
    #     model.fit(X=X_poly_train, y=Y_train)
    #     y_pred = model.predict(X_poly_test)
    #     r2 = r2_score(y_true=Y_test, y_pred=y_pred)
    #     r2 = np.round(r2, decimals=2)
    #     rmse = np.sqrt(mean_squared_error(y_true=Y_test, y_pred=y_pred))
    #     rmse = np.round(rmse, decimals=2)
    #     # print("r2 : {}, rmse :{}\n".format(r2, rmse))
    #     r2_list.append(r2)
    #     rmse_list.append(rmse)
    # print("\nR2_Raw: {}, R2_mean: {:.2f}, R2_Std: {:.2f}".format(r2_list, np.mean(r2_list), np.std(r2_list)))
    # print("\nRMSE_Raw: {}, RMSE_mean: {}, RMSE_Std: {}".format(rmse_list, np.mean(rmse_list), np.std(rmse_list)))
    ########################################################################
    # split data and fit model
    # training_fraction = 0.80
    # # X_train_poly = poly_features.fit_transform(X_train)
    # X_train_poly, X_test_poly = split_train_test_data(X_poly, training_fraction=training_fraction)
    # print("\ntrain points: {}, test points: {}\n".format(len(X_train_poly), len(X_test_poly)))
    #
    # Y_train, Y_test = split_train_test_data(Y.values, training_fraction=training_fraction)
    # model = LinearRegression()
    # model.fit(X_train_poly, Y_train)
    #
    # # test model
    # # X_test_poly = poly_features.fit_transform(X_test)
    # y_pred = model.predict(X_test_poly)
    #
    # r2 = r2_score(Y_test, y_pred)
    # r2 = np.round(r2, decimals=2)
    # rmse = np.sqrt(mean_squared_error(Y_test, y_pred))
    # rmse = np.round(rmse, decimals=2)
    # print("r2 : {}, rmse :{}\n".format(r2, rmse))


if __name__ == "__main__":
    DEBUG = True
    img_dir = "../../../dataset/PETS_1/JPEGImages"
    annot_dir = "../../../dataset/PETS_1/Annotations"

    # ref_cam = 7
    # collab_cams = [8,6,5]
    src_cam = 57
    dst_cam = 75

    src_points, dst_points = [], []

    # create training & test frame names
    # random_frame_sampling()

    # read training frame names
    trainval_frames = np.loadtxt("../../../dataset/PETS_1/ImageSets/trainval.txt", dtype=np.int32)

    # for i in range(0, 794):
    for i in trainval_frames:
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
                    # s_width = int(s_xmax) - int(s_xmin)
                    # s_height = int(s_ymax) - int(s_ymin)
                    # s_mid_x = int(s_xmin) + s_width / 2
                    # src_points.append([s_mid_x, s_ymax, s_width, s_height])
                    src_points.append([s_xmin, s_ymin, s_xmax, s_ymax])
                    # src_points.append([s_xmin, s_ymax])
                    # src_points.append([s_xmax, s_ymax])

                    # d_width = int(d_xmax) - int(d_xmin)
                    # d_height = int(d_ymax) - int(d_ymin)
                    # d_mid_x = int(d_xmin) + d_width / 2
                    dst_points.append([d_xmin, d_ymin, d_xmax, d_ymax])
                    # dst_points.append([d_mid_x, d_ymax, d_width, d_height])
                    # dst_points.append([d_xmin, d_ymax])
                    # dst_points.append([d_xmax, d_ymax])
                    break

    # fit linear regresssion
    # fit_linear_regression_bt_pts(src_points=src_points, dst_points=dst_points)

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

    # visualize_mid_points(s_points=src_points, d_points=dst_points)
    # fit_lin_reg_bt_pts_height(s_points=src_points, d_points=dst_points)
    # fit_fundamental_mat(s_points=src_points, d_points=dst_points)
    print("\ntotal points: {}\n".format(len(src_points)))
    fit_poly_feature_linear_reg_bt_pt_height_width(s_points=src_points, d_points=dst_points)
