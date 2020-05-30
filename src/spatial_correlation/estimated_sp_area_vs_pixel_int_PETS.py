"""
module to estimate resolution reduction from collaboration. Energy savings section in dissertation.
"""

import cv2
import numpy as np

if __name__ == "__main__":
    PXL_INTENSITY_TH = 170
    INPUT_DIR = r"performance_analysis/spatial_area_estimation_PETS/"
    ref_cam = 7
    collab_cam = 8

    gt_overlap_coords_7_5 = (21, 100, 571, 493)
    gt_overlap_coords_7_6 = (67, 85, 585, 470)
    gt_overlap_coords_7_8 = (28, 101, 617, 492)

    image_name = "cam_{}_{}_area.jpg".format(ref_cam, collab_cam)
    frame = cv2.imread("{}/{}".format(INPUT_DIR, image_name))
    assert frame is not None

    # select all pixels with intensities below max_pixel_intensity
    desired_coordinates = []
    # select all pixels below this intensity value
    frame_copy = frame.copy()
    frame = frame[:, :, 0]
    frame_shape = frame.shape
    for r, c in np.ndindex(frame_shape):
        if frame[r, c] <= PXL_INTENSITY_TH:
            desired_coordinates.append((c, r))  # x,y

    # estimate enclosing rectangle for all these points
    min_x = min(desired_coordinates, key=lambda x: x[0])[0]
    min_y = min(desired_coordinates, key=lambda x: x[1])[1]
    max_x = max(desired_coordinates, key=lambda x: x[0])[0]
    max_y = max(desired_coordinates, key=lambda x: x[1])[1]
    print([min_x, min_y, max_x, max_y])
    print("width; {}, height: {}".format(max_x - min_x, max_y - min_y))

    est_area_global = [min_x, min_y, max_x, max_y]
    cv2.rectangle(frame_copy, (min_x, min_y), (max_x, max_y), (0, 0, 255), 2)
    if collab_cam == 5:
        gt_overlap_coords = gt_overlap_coords_7_5
    elif collab_cam == 6:
        gt_overlap_coords = gt_overlap_coords_7_6
    elif collab_cam == 8:
        gt_overlap_coords = gt_overlap_coords_7_8

    # cv2.rectangle(frame_copy, (gt_overlap_coords[0], gt_overlap_coords[1]),
    #               (gt_overlap_coords[2], gt_overlap_coords[3]), (0, 255, 0), 2)
    # draw black border around teh image
    cv2.rectangle(frame_copy, (0, 0), (720, 576), (0, 0, 0), 2)

    cv2.imwrite("estimated_area_c{}_{}_PETS.jpg".format(ref_cam, collab_cam), frame_copy)
    cv2.waitKey(-1)
