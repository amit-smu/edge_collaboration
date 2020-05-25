"""
module to check for spatial coorrelation between various camera frames
"""

import cv2
import numpy as np
from src.client import detect_person
from src.obj_comp_utils import compare_hist, compare_sift, compare_hu_moments
import pandas as pd

# global variables
MOTION_THRESHOLD = 12


def format_object(bboxes, org_img):
    r, c, _ = org_img.shape
    f_boxes = []
    for box in bboxes:
        classname = classes[int(box[0])]
        if classname != "person":
            continue
        confidence = np.around(box[1] * 100, decimals=1)
        xmin = int(box[2] * c / scaled_img_dims[0])
        ymin = int(box[3] * r / scaled_img_dims[1])
        xmax = int(box[4] * c / scaled_img_dims[0])
        ymax = int(box[5] * r / scaled_img_dims[1])
        f_boxes.append([classname, confidence, xmin, ymin, xmax, ymax])

    return f_boxes


def compare_objects(frame_num, bbox_list_1, bbox_list_2, frame_1, frame_2, view_1, view_2):
    """
    compare two list of given objects (bounding boxes) in terms of how similar are they using methods
    in object compare utility module
    :param obj_1:
    :param obj_2:
    :return:
    """
    output = []
    for box_1 in bbox_list_1:
        obj_1 = frame_1[box_1[3]:box_1[5], box_1[2]:box_1[4]]
        if DEBUGGING:
            cv2.imshow("obj_1", obj_1)
        for box_2 in bbox_list_2:
            obj_2 = frame_2[box_2[3]:box_2[5], box_2[2]:box_2[4]]
            if DEBUGGING:
                cv2.imshow("obj_2", obj_2)
            color_score = 1 - compare_hist(obj_1, obj_2)
            sift_score = compare_sift(obj_1, obj_2)
            hu_score = compare_hu_moments(obj_1, obj_2)
            output.append(
                [frame_num, view_1, view_2, box_1, box_2, color_score, sift_score, color_score + sift_score, hu_score])

    return output


def extract_contour(contour, image):
    print("")


def op_flow_from_view(f_num, next_frame_num, view_dir):
    prev_frame_name = "{}{}.jpg".format("frame", f_num)
    next_frame_name = "{}{}.jpg".format("frame", next_frame_num)
    previous_frame = cv2.imread("{}/{}".format(view_dir, prev_frame_name))
    previous_frame_color = previous_frame.copy()
    previous_frame = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
    next_frame = cv2.imread("{}/{}".format(view_dir, next_frame_name))
    next_frame = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(prev=previous_frame, next=next_frame, flow=None, pyr_scale=0.5,
                                        levels=3, winsize=10, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)

    # cartesian to polar conversion
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    # ang = ang * (180 / np.pi)
    mag_copy = mag.copy()
    mag_copy[mag_copy < MOTION_THRESHOLD] = 0
    mag_copy[mag_copy >= MOTION_THRESHOLD] = 1
    mag_copy = np.array(mag_copy, dtype="uint8")
    previous_frame_flow = cv2.bitwise_and(previous_frame, previous_frame, mask=mag_copy)

    # morph ops
    kernel = np.ones((3, 3), np.uint8)
    edges_dilated = cv2.dilate(previous_frame_flow, kernel, iterations=5)
    # cv2.imshow("edges_dilated", edges_dilated)
    edges_dilated = cv2.erode(edges_dilated, kernel, iterations=5)
    cv2.imshow("previous_frame_flow", edges_dilated)

    # contour detection
    _, contours, hierarchy = cv2.findContours(edges_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_contours = []
    for c in contours:
        area = cv2.contourArea(c)
        print("Contour area :{}".format(area))
        if area >= 3000:
            final_contours.append(c)
    cv2.drawContours(previous_frame_color, final_contours, -1, (0, 255, 0), 2)
    cv2.imshow("previous_frame_color", previous_frame_color)

    # # blob detector
    # # Setup SimpleBlobDetector parameters.
    # params = cv2.SimpleBlobDetector_Params()
    #
    # # Change thresholds
    # params.minThreshold = 10
    # params.maxThreshold = 200
    #
    # # Filter by Area.
    # params.filterByArea = True
    # params.minArea = 1500
    #
    # # Filter by Circularity
    # params.filterByCircularity = False
    # params.minCircularity = 0.1
    #
    # # Filter by Convexity
    # params.filterByConvexity = False
    # params.minConvexity = 0.87
    #
    # # Filter by Inertia
    # params.filterByInertia = False
    # params.minInertiaRatio = 0.01
    #
    # blob_detector = cv2.SimpleBlobDetector_create(params)
    # blobs = blob_detector.detect(edges_dilated)
    # blobs_img = cv2.drawKeypoints(previous_frame_color, blobs, np.array([]), (0, 255, 0))
    # cv2.imshow("blobs", blobs_img)
    cv2.waitKey(-1)


def mv_based_correlation(f_numbers, view_5_dir, view_7_dir, view_8_dir):
    """
    use motion
    :param f_numbers:
    :param view_5_dir:
    :param view_7_dir:
    :param view_8_dir:
    :return:
    """

    for f_num in f_numbers:
        if f_num == "_0030":
            # mv_frame_numbers = ["_0031", "_0032", "_0033", "_0034", "_0035"]
            next_frame_num = "_0031"
        elif f_num == "_0128":
            next_frame_num = "_0129"
        elif f_num == "_0470":
            next_frame_num = "_0471"
        elif f_num == "_0700":
            next_frame_num = "_0701"
        elif f_num == "_0752":
            next_frame_num = "_0753"

        # find optical flow from view 5
        op_flow_from_view(f_num, next_frame_num, view_dir=view_8_dir)


if __name__ == "__main__":
    DEBUGGING = True
    output_csv_file = "../analysis/sp_corr_comp_scores.csv"
    scaled_img_dims = (300, 300)  # size that object detector uses for input image
    data_dir = "../../dataset"
    view_5_dir = "{}/{}".format(data_dir, "View_005")
    view_7_dir = "{}/{}".format(data_dir, "View_007")
    view_8_dir = "{}/{}".format(data_dir, "View_008")
    classes = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
               'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor']

    f_numbers = ["_0030", "_0128", "_0470", "_0700", "_0752"]
    mv_based_correlation(f_numbers=f_numbers, view_5_dir=view_5_dir, view_7_dir=view_7_dir, view_8_dir=view_8_dir)
    frame_output_list = []
    for f_num in f_numbers:
        # f_num = "0030"
        print("f_num", f_num)
        frame_name = "{}{}.jpg".format("frame", f_num)
        frame_v5 = cv2.imread("{}/{}".format(view_5_dir, frame_name))
        frame_v7 = cv2.imread("{}/{}".format(view_7_dir, frame_name))
        frame_v8 = cv2.imread("{}/{}".format(view_8_dir, frame_name))

        cv2.imshow("frame_v5", frame_v5)
        cv2.imshow("frame_v7", frame_v7)
        cv2.imshow("frame_v8", frame_v8)
        cv2.waitKey(100)

        # get bounding boxes from object detector
        bboxes = detect_person("{}/{}".format(view_5_dir, frame_name))
        bboxes_view_5 = format_object(bboxes, frame_v5)

        bboxes = detect_person("{}/{}".format(view_7_dir, frame_name))
        bboxes_view_7 = format_object(bboxes, frame_v7)

        bboxes = detect_person("{}/{}".format(view_8_dir, frame_name))
        bboxes_view_8 = format_object(bboxes, frame_v8)

        # match detected objects using color-hist and sift features

        # compare view 5 & 7
        temp = compare_objects(f_num, bboxes_view_5, bboxes_view_7, frame_v5, frame_v7, 5, 7)
        frame_output_list.extend(temp)

        # compare view 5 & 8
        temp = compare_objects(f_num, bboxes_view_5, bboxes_view_8, frame_v5, frame_v8, 5, 8)
        frame_output_list.extend(temp)

        # compare view 7 & 8
        temp = compare_objects(f_num, bboxes_view_7, bboxes_view_8, frame_v7, frame_v8, 7, 8)
        frame_output_list.extend(temp)

    # write to csv file
    df = pd.DataFrame(data=frame_output_list, index=None,
                      columns=['frame_num', 'view_1', 'view_2', 'obj_box_1', 'obj_box_2', 'color_score', 'sift_score',
                               'combined_score', 'hu_score'])
    df['frame_num'] = df['frame_num'].astype(dtype=object)
    df.to_csv(output_csv_file)
