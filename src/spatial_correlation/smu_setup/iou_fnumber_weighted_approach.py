"""
file used for plotting variation of IoU score (between estimated and actual overlap area) vs total frames analysed
for weighted-approach
"""
# import utils
import cv2
import os
import xml.dom.minidom
import numpy as np


def get_bb_coords(xml_path):
    dom_tree = xml.dom.minidom.parse(xml_path)
    collection = dom_tree.documentElement
    xmin = collection.getElementsByTagName("xmin")[0].firstChild.nodeValue
    ymin = collection.getElementsByTagName("ymin")[0].firstChild.nodeValue
    xmax = collection.getElementsByTagName("xmax")[0].firstChild.nodeValue
    ymax = collection.getElementsByTagName("ymax")[0].firstChild.nodeValue
    return [int(xmin), int(ymin), int(xmax), int(ymax)]


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


def get_gt_sp_overlap_coordinates(view_1, view_2):
    """
    retrieves the coordinates of bounding box representing the marked spatial overlap between view 1 and view 2
    :param view_1:
    :param view_2:
    :return:
    """
    xml_file_name = "frame_000000_{:02d}_{:02d}.xml".format(view_1, view_2)
    sp_overlap_dir = "../../../dataset/spatial_overlap/WILDTRACK/"
    xml_path = "{}/{}".format(sp_overlap_dir, xml_file_name)
    return get_bb_coords(xml_path=xml_path)


def get_key(x):
    x = x[:-4].split("_")[-1]
    return int(x)


if __name__ == "__main__":
    dir_name = "../../../rpi_hardware/raw_image_processing/data/episode_1/"
    ref_cam = 2
    collab_cam = 1
    # gt_box_coords_vw_1 = get_gt_sp_overlap_coordinates(ref_cam, collab_cam)
    # gt_box_coords_vw_1 = [360, 0, 990, 1080]
    gt_box_coords_vw_1 = [360, 0, 1010, 1080]
    max_pixel_intensity = 170

    scale = 1056 / 1080
    gt_box_coords_vw_1 = [int(t * scale) for t in gt_box_coords_vw_1]

    iou_scores = []
    est_area_global = []

    frame_list = os.listdir("{}/intermediate_frames/cam_{}_{}".format(dir_name, ref_cam, collab_cam))
    frame_list = sorted(frame_list, key=lambda x: get_key(x))
    # frame_list = np.arange(0, 2000, 5)
    for f_num in frame_list:
        # f_name = "marked_area_f_{}.jpg".format(f_num)
        print(f_num)
        frame_path = "{}/{}/cam_{}_{}/{}".format(dir_name, "intermediate_frames", ref_cam, collab_cam, f_num)
        # print(frame_path)

        if not os.path.exists(frame_path):
            print("path doesn't exist: {}".format(frame_path))
            continue
        frame = cv2.imread(frame_path)
        assert frame is not None

        # select all pixels with intensities below max_pixel_intensity
        desired_coordinates = []
        # select all pixels below this intensity value
        frame_copy = frame.copy()
        frame = frame[:, :, 0]
        frame_shape = frame.shape
        for r, c in np.ndindex(frame_shape):
            if frame[r, c] <= max_pixel_intensity:
                desired_coordinates.append((c, r))  # x,y

        # estimate enclosing rectangle for all these points
        if len(desired_coordinates) == 0:
            iou_scores.append(0)
            print(iou_scores)
            continue
        min_x = min(desired_coordinates, key=lambda x: x[0])[0]
        min_y = min(desired_coordinates, key=lambda x: x[1])[1]
        max_x = max(desired_coordinates, key=lambda x: x[0])[0]
        max_y = max(desired_coordinates, key=lambda x: x[1])[1]

        est_area_global = [min_x, min_y, max_x, max_y]
        # # merge the estimated area with global area
        # if len(est_area_global) > 0:
        #     min_x_global = min(min_x, est_area_global[0])
        #     min_y_global = min(min_y, est_area_global[1])
        #     max_x_global = max(max_x, est_area_global[2])
        #     max_y_global = max(max_y, est_area_global[3])
        # else:
        #     est_area_global = [min_x, min_y, max_x, max_y]

        i_score = bb_iou(gt_box_coords_vw_1, est_area_global)
        iou_scores.append(i_score)

        print("iou_score : {}".format(i_score))

        if len(est_area_global) > 0:
            cv2.rectangle(frame_copy, (est_area_global[0], est_area_global[1]),
                          (est_area_global[2], est_area_global[3]),
                          (255, 0, 0), 2)
            cv2.imwrite("final_area.jpg", frame_copy)
        # break
    with open("{}/overlap_vs_pixel_int_{}.txt".format(dir_name, max_pixel_intensity), 'w') as out:
        for i in iou_scores:
            print(i)
            out.write("{}\n".format(i))

        # cv2.waitKey(-1)
