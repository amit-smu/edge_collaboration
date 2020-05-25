"""
module to create training data images for training a ssd model. It will take 2 images as input and
will give bounding boxes of people in those images
Images selected from PETS dataset by cropping the overlapping area between two camera sensors.
"""

import cv2
import numpy as np
import pandas as pd


def create_frame_name(f_num):
    """
    crearte frame name from frame number
    :param f_num:
    :return:
    """
    if f_num < 10:
        return "frame_000{}".format(f_num)
    elif f_num < 100:
        return "frame_00{}".format(f_num)
    elif f_num < 1000:
        return "frame_0{}".format(f_num)


if __name__ == "__main__":
    dir_name = "../../dataset"

    view_1_number = 7
    view_2_number = 8
    view_1_name = "View_00{}".format(view_1_number)
    view_2_name = "View_00{}".format(view_2_number)

    # output directory
    output_dir = "{}/model_training_data_pets/View_00{}_00{}".format(dir_name, view_1_number, view_2_number)

    image_size = (720, 576)  # PETS dataset (width, height)

    # estimated spatial_overlap
    est_sp_overlap_vw_1 = (100, 100, 300, 450)  # overlap coordinates (x1,y1,x2,y2) between vw1 and vw2 projected on vw1
    est_sp_overlap_vw_2 = (100, 100, 300, 450)

    # ground truth spatial overlap area coordinates
    # gt_box_coords_vw_1 = [100, 200, 300, 350]  # [x1,y1,x2,y2]
    # gt_box_coords_vw_1 = get_gt_sp_overlap_coordinates("0{}".format(view_1_number), "0{}".format(view_2_number))
    # gt_box_coords_vw_2 = get_gt_sp_overlap_coordinates("0{}".format(view_2_number), "0{}".format(view_1_number))

    view_1_grnd_truth = pd.read_csv("{}/{}.txt".format(dir_name, view_1_name), delimiter=" ")
    view_2_grnd_truth = pd.read_csv("{}/{}.txt".format(dir_name, view_2_name), delimiter=" ")

    for frame_number in range(100, 350):
        print("frame number : {}".format(frame_number))
        frame_name = create_frame_name(frame_number)

        view_1_frame = cv2.imread("{}/{}/{}.jpg".format(dir_name, view_1_name, frame_name))
        view_2_frame = cv2.imread("{}/{}/{}.jpg".format(dir_name, view_2_name, frame_name))

        # get objects from frames (using ground truth)
        view_1_objects = view_1_grnd_truth[view_1_grnd_truth['frame_number'] == frame_number]
        view_2_objects = view_2_grnd_truth[view_2_grnd_truth['frame_number'] == frame_number]
        # apply filters on objects
        view_1_objects = view_1_objects[(view_1_objects['lost'] == 0) & (view_1_objects['occluded'] == 0)]
        view_2_objects = view_2_objects[(view_2_objects['lost'] == 0) & (view_2_objects['occluded'] == 0)]

        # name of processed frames
        vw1_frame_name = "{}_{}.jpg".format(view_1_number, frame_name)
        vw2_frame_name = "{}_{}.jpg".format(view_2_number, frame_name)

        # crop views and write to output files
        cropped_view_1 = view_1_frame[est_sp_overlap_vw_1[1]:est_sp_overlap_vw_1[3],
                         est_sp_overlap_vw_1[0]:est_sp_overlap_vw_1[2]]

        cropped_view_2 = view_2_frame[est_sp_overlap_vw_2[1]:est_sp_overlap_vw_2[3],
                         est_sp_overlap_vw_2[0]:est_sp_overlap_vw_2[2]]

        cv2.imwrite("{}/{}".format(output_dir, vw1_frame_name), cropped_view_1)
        cv2.imwrite("{}/{}".format(output_dir, vw2_frame_name), cropped_view_2)
        print("")