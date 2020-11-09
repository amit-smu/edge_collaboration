"""
file used for plotting variation of IoU score (between estimated and actual overlap area) vs total frames analysed
for weighted-approach
"""
import utils
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


if __name__ == "__main__":
    ref_cam = 4
    collab_cam = 7
    gt_box_coords_vw_1 = get_gt_sp_overlap_coordinates(ref_cam, collab_cam)
    max_pixel_intensity = 170

    iou_scores = []
    est_area_global = []

    # frame_list = os.listdir("intermediate_frames/")
    # frame_list = sorted(frame_list)
    frame_list = np.arange(0, 2000, 5)
    for f_num in frame_list:
        f_name = "marked_area_cam_r{}_c{}_f_{}.jpg".format(ref_cam, collab_cam, f_num)
        # print(f_name)
        frame_path = "{}/{}".format("intermediate_frames", f_name)
        print(frame_path)

        if not os.path.exists(frame_path):
            print("path doesn't exist")
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

        i_score = utils.bb_iou(gt_box_coords_vw_1, est_area_global)
        iou_scores.append(i_score)

        print("iou_score : {}".format(i_score))

        if len(est_area_global) > 0:
            cv2.rectangle(frame_copy, (est_area_global[0], est_area_global[1]),
                          (est_area_global[2], est_area_global[3]),
                          (255, 0, 0), 2)
            cv2.imwrite("final_area.jpg", frame_copy)

        # break
    for i in iou_scores:
        print(i)

        # cv2.waitKey(-1)
