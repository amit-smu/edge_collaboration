"""
module to test performance of polynomial feature based linear regression for wildtrack dataset
"""
import numpy as np

import pickle
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from matplotlib import pyplot as plt


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


def plot_regression_results(iou_values):
    iou_ticks = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    iou_fractions = []
    for tick in iou_ticks:
        # find fraction of iou values above this iou
        target_iou = [i for i in iou_values if i >= tick]
        total_iou = len(iou_values)
        fraction = len(target_iou) / total_iou
        iou_fractions.append(np.round(fraction * 100.0, decimals=2))
    print(iou_ticks)
    for t in iou_fractions:
        print(t)


def test_poly_feature_linear_reg_bt_pt_height_width(s_points, d_points):
    """
    test polynomial feature linear regression model.
    Metric: IoU score with ground truth & predicted bounding boxes
    :param s_points:
    :param d_points:
    :return:
    """

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

    df = df.astype('int32')
    X = df[['src_bt_mid_x', 'src_ymax', 'src_width', 'src_height']]
    Y = df[['dst_bt_mid_x', 'dst_ymax', 'dst_width', 'dst_height']]
    # convert to 2d np array
    X = X.values
    Y = Y.values

    poly_features = PolynomialFeatures(degree=degree, interaction_only=False)
    X_poly = poly_features.fit_transform(X)

    # #################### LOAD REGRESSION MODEL ######################################
    reg_model = None
    # model_file_path = "regression_models/poly_feature_linear_regression_deg_{}_interaction_false_cam_{}".format(degree,
    #                                                                                                             src_cam)
    model_file_path = "regression_models/poly_feature_l_reg_deg_{}_inter_false_src_{}_dst_{}_full_img".format(degree,
                                                                                                              src_cam,
                                                                                                              dst_cam)
    print("degree: {}, src_cam: {}\n".format(degree, src_cam))
    with open(model_file_path, 'rb') as input_file:
        reg_model = pickle.load(input_file)

    # ##################### COMPUTE IoU SCORES ########################################
    iou_scores = []
    for index, row in enumerate(X_poly):
        row = np.reshape(row, newshape=(1, -1))
        row_pred = reg_model.predict(row)
        row_pred = np.array(row_pred, dtype=np.int32)
        # score = compute_iou_score(row_pred[0], actual_bbox=Y[index])
        score = compute_iou_score(pred_coords=row_pred[0], actual_coords=Y[index])

        iou_scores.append(score)
    plot_regression_results(iou_scores)
    # plt.figure()
    # plt.title(src_cam)
    # plt.scatter(x=np.arange(0, len(iou_scores)), y=iou_scores)
    plt.show()


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
    DEBUG = False

    image_size = (1920, 1080)  # WILDTRACK dataset (width, height)
    dir_name = "../../../dataset/Wildtrack_dataset"
    img_dir = "../../../dataset/Wildtrack_dataset/Image_subsets"
    annot_dir = "../../../dataset/Wildtrack_dataset/annotations_positions"

    src_cam = 7
    dst_cam = 5
    degree = 5

    src_points, dst_points = [], []

    # read training frame names
    # trainval_frames = np.loadtxt("{}/ImageSets/trainval.txt".format(dir_name), dtype=np.int32)
    test_frames = np.loadtxt("{}/ImageSets/test.txt".format(dir_name), dtype=np.int32)
    print("test_frames : {}".format(len(test_frames)))
    for i in test_frames:
        src_annot_name = "{:08d}.json".format(i)
        # print("frame : {}".format(src_annot_name))
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
    test_poly_feature_linear_reg_bt_pt_height_width(s_points=src_points, d_points=dst_points)
