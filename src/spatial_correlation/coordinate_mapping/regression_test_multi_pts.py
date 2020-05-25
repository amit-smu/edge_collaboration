import cv2
import os
import numpy as np
from xml.etree import ElementTree
import pickle


def test_linear_reg_bt_pt_diagonal(s_points, d_img, obj_id):
    """
    use bottom mid point and diagonal
    :param d_img:
    :param s_points:
    :return:
    """
    xmin, ymin, xmax, ymax = s_points
    width = xmax - xmin
    diagonal = ((xmax - xmin) ** 2 + (ymax - ymin) ** 2) ** 0.5
    mid_x = int(xmin + width / 2)

    mid_x_t, ymax_t, diagonal_t = regressor.predict([[mid_x, ymax, diagonal]])[0]

    mid_x_t = int(mid_x_t)
    ymax_t = int(ymax_t)
    diagonal_t = int(diagonal_t)

    cv2.circle(d_img, (mid_x_t, ymax_t), 4, (0, 255, 0), 2)
    cv2.putText(dst_img, "{}".format(obj_id), (mid_x_t, ymax_t - 3), cv2.FONT_HERSHEY_PLAIN, 1.2,
                (0, 255, 0), 1)


def test_linear_reg_bt_pt_wid_hei(s_points, d_img, obj_id):
    """
    use bottom mid point, width, height
    :param d_img:
    :param s_points:
    :return:
    """
    xmin, ymin, xmax, ymax = s_points

    width = xmax - xmin
    height = ymax - ymin
    mid_x = int(xmin + width / 2)

    mid_x_t, ymax_t, width_t, height_t = regressor.predict([[mid_x, ymax, width, height]])[0]
    mid_x_t = int(mid_x_t)
    ymax_t = int(ymax_t)
    width_t = int(width_t)
    height_t = int(height_t)
    # xmin_t, ymax_t, xmax_t, ymax_t = regressor.predict([[xmin, ymax, xmax, ymax]])[0]

    # compute bounding box
    org_height = ymax - ymin
    org_width = (xmax - xmin)
    bb_xmin = int(mid_x_t - width_t / 2)
    bb_xmax = int(mid_x_t + width_t / 2)
    bb_ymin = ymax_t - height_t
    bb_ymax = ymax_t
    # bb_height = int(1.5 * (xmax_t - xmin_t))
    # bb_xmin = int(xmin_t)
    # bb_ymin = int(ymax_t - bb_height)
    # bb_xmax = int(xmax_t)
    # bb_ymax = int(ymax_t)

    # cv2.circle(dst_img, (mid_x_t, ymax_t), 4, (0, 255, 0), 2)
    # cv2.putText(dst_img, "{}".format(obj_id), (mid_x_t, ymax_t - 3), cv2.FONT_HERSHEY_PLAIN, 1.2,
    #             (0, 255, 0), 1)
    cv2.rectangle(d_img, (bb_xmin, bb_ymin), (bb_xmax, bb_ymax), (0, 255, 255), 1)
    cv2.putText(d_img, "{}".format(obj_id), (bb_xmin, bb_ymin - 3), cv2.FONT_HERSHEY_PLAIN, 1.2,
                (0, 255, 0), 1)


def test_lin_reg_bt_pts_height(s_points, d_img, obj_id):
    """
    use both bottom points, height
    :param d_img:
    :param s_points:
    :return:
    """
    xmin, ymin, xmax, ymax = s_points
    height = ymax - ymin
    xmin_t, ymax_t, xmax_t, ymax_t1, height_t = regressor.predict([[xmin, ymax, xmax, ymax, height]])[0]

    xmin_t = int(xmin_t)
    ymax_t = int(ymax_t)
    xmax_t = int(xmax_t)
    ymax_t1 = int(ymax_t1)
    height_t = int(height_t)

    # compute bounding box
    bb_xmin = xmin_t
    bb_ymin = ymax_t - height_t
    bb_xmax = xmax_t
    bb_ymax = ymax_t1

    cv2.rectangle(d_img, (bb_xmin, bb_ymin), (bb_xmax, bb_ymax), (0, 255, 255), 1)
    cv2.putText(d_img, "{}".format(obj_id), (bb_xmin, bb_ymin - 3), cv2.FONT_HERSHEY_PLAIN, 1.2,
                (0, 255, 0), 1)


def test_fundamental_matrix(s_points, d_img, obj_id):
    pt1 = [s_points[0], s_points[1]]
    pt1 = np.reshape(pt1, (1, 2))
    _, c, _ = d_img.shape
    r = cv2.computeCorrespondEpilines(pt1, 1, F=f_mat)
    r = r.ravel()
    x0, y0 = map(int, [0, -r[2] / r[1]])
    x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
    # cv2.line(d_img, (x0, y0), (x1, y1), (0, 255, 0), 2)

    # pt2 = [s_points[2], s_points[3]]
    mid_bt_x = s_points[0] + (s_points[2] - s_points[0]) / 2
    mid_bt_x = int(mid_bt_x)
    pt2 = [mid_bt_x, s_points[3]]

    pt2 = np.reshape(pt2, (1, 2))
    r = cv2.computeCorrespondEpilines(pt2, 1, F=f_mat)
    r = r.ravel()

    x0, y0 = map(int, [0, -r[2] / r[1]])
    x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
    cv2.line(d_img, (x0, y0), (x1, y1), (255, 255, 0), 2)


if __name__ == "__main__":
    DEBUG = False
    img_dir = "../../../dataset/PETS_1/JPEGImages"
    annot_dir = "../../../dataset/PETS_1/Annotations"

    src_cam = 67
    dst_cam = 78

    # coordinate_mapping = None
    # with open("homography_87_bottom_pts", "rb") as input_file:
    #     coordinate_mapping = pickle.load(input_file)
    #
    # print("coordinate_mapping : {}\n".format(coordinate_mapping))

    # load regressor
    regressor = None
    # regressor_name = "regression_{}_multi_pts".format(src_cam)
    regressor_name = "regression_{}_bt_pts_height".format(src_cam)

    with open(regressor_name, 'rb') as input_f:
        regressor = pickle.load(input_f)
        assert regressor is not None
        print("regression model loaded.. :{}\n".format(regressor_name))

    # load fundamental matrix
    f_mat = [[1.98167825e-06, -7.18564454e-05, - 2.71008140e-03],
             [-1.32202843e-04, 8.20580299e-05, 1.19213906e-01],
             [-2.20905096e-03, -7.45274374e-02, 1.00000000e+00]]
    f_mat = np.float64(f_mat)

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
            height = ymax - ymin
            mid_x = int(xmin + width / 2)
            cv2.circle(src_img, (mid_x, ymax), 4, (0, 0, 255), 2)
            cv2.putText(src_img, "{}".format(obj_id), (mid_x, ymax - 3), cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 255, 255),
                        2)

            # translate points to dst imgage
            # test_linear_reg_bt_pt_wid_hei([xmin, ymin, xmax, ymax], dst_img, obj_id)
            # test_lin_reg_bt_pts_height([xmin, ymin, xmax, ymax], dst_img, obj_id)
            test_fundamental_matrix([xmin, ymin, xmax, ymax], dst_img, obj_id)
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
