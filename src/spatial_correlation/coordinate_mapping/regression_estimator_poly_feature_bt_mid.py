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
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error


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
