"""
regression estimator for coordinate mapping from one frame to another. Linear regression
use multiple points from each bounding box. These points are on the road.
Final file used for regression (obsolete -- optimized version has replaced this one)
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
        src_img_name = "C{}/{:08d}.png".format(src_cam, 0)
        src_img_path = "{}/{}".format(img_dir, src_img_name)
        src_img = cv2.imread(src_img_path)

        dst_img_name = "C{}/{:08d}.png".format(dst_cam, 0)
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
        cv2.waitKey(-1)

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
    # degree = 2
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
    model_file_path = "regression_models/poly_feature_l_reg_deg_{}_inter_false_src_{}_dst_{}_full_img".format(degree,
                                                                                                              src_cam,
                                                                                                              dst_cam)
    with open(model_file_path, 'wb') as output:
        pickle.dump(best_model, output)
        print("best_rmse : {}".format(min_rmse))


def check_coordinates_for_error(s_view):
    non_erroneous_coords = True
    x1, y1, x2, y2 = s_view["xmin"], s_view["ymin"], s_view["xmax"], s_view["ymax"]
    if x1 < 0 or x2 < 0 or y1 < 0 or y2 < 0:
        non_erroneous_coords = False

    elif x1 >= image_size[0] or x2 >= image_size[0] or y1 >= image_size[1] or y2 >= image_size[1]:
        # ignore coordinates bigger than image size
        non_erroneous_coords = False

    return non_erroneous_coords


if __name__ == "__main__":
    DEBUG = True

    image_size = (1920, 1080)  # WILDTRACK dataset (width, height)
    dir_name = "../../../dataset/Wildtrack_dataset"
    img_dir = "../../../dataset/Wildtrack_dataset/Image_subsets"
    annot_dir = "../../../dataset/Wildtrack_dataset/annotations_positions"

    src_cam = 6
    dst_cam = 1
    degree = 4

    src_points, dst_points = [], []

    # read training frame names
    trainval_frames = np.loadtxt("{}/ImageSets/trainval.txt".format(dir_name), dtype=np.int32)

    for i in trainval_frames:
        src_annot_name = "{:08d}.json".format(i)
        print("frame : {}".format(src_annot_name))
        src_annot_path = "{}/{}".format(annot_dir, src_annot_name)
        gt_data = pd.read_json(src_annot_path)

        for person in gt_data.iterrows():
            person = person[1]
            person_id = person["personID"]
            views = person["views"]
            selected_views = []
            for v in views:
                v_num = v['viewNum']
                if v_num + 1 == src_cam or v_num + 1 == dst_cam:  # starts from 0
                    selected_views.append(v)

            # check person coordinates for <0 values
            if check_coordinates_for_error(selected_views[0]) and check_coordinates_for_error(selected_views[1]):
                # add source view to list
                src_view = [view for view in selected_views if view["viewNum"] + 1 == src_cam][0]
                src_points.append([src_view['xmin'], src_view['ymin'], src_view['xmax'], src_view['ymax']])
                # add destination view to list
                dst_view = [view for view in selected_views if view["viewNum"] + 1 == dst_cam][0]
                dst_points.append([dst_view['xmin'], dst_view['ymin'], dst_view['xmax'], dst_view['ymax']])
            else:
                continue
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
