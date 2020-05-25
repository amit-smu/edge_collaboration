"""
module to test the performance, in terms of IoU score, of selecting overlap area with varying pixel intensity values.
FOr a given pixel intensity threshold, select all the pixels under that intensity and estimate overlap area, then find
IoU score of this area with actual overlap area.
"""
import utils
import numpy as np
import cv2
from matplotlib import pyplot as plt


def weighted_spatial_overlap_performance(masked_image, gt_overlap_area):
    """
    how does estimated overlap IoU varies with various values of pixel intensity threshold
    :param masked_image: The estimated overlap area
    :param gt_overlap_area: coordinates of ground truth of overlap area
    :return:
    """
    print()
    pixel_intensities = [245, 230, 210, 190, 170, 150, 130, 110, 90, 70, 50, 30, 20, 10, 0]
    iou_scores = []  # iou score for each pixel intensity

    masked_image_org = masked_image.copy()

    masked_image = masked_image[:, :, 0]  # drop 3rd dimension(depth)
    image_shape = masked_image.shape
    for intensity_threshold in pixel_intensities:
        desired_coordinates = []
        # select all pixels below this intensity value
        for r, c in np.ndindex(image_shape):
            if masked_image[r, c] <= intensity_threshold:
                desired_coordinates.append((c, r))  # x,y

        # estimate enclosing rectangle for all these points
        min_x = min(desired_coordinates, key=lambda x: x[0])[0]
        min_y = min(desired_coordinates, key=lambda x: x[1])[1]
        max_x = max(desired_coordinates, key=lambda x: x[0])[0]
        max_y = max(desired_coordinates, key=lambda x: x[1])[1]
        iou_score = utils.bb_iou([min_x, min_y, max_x, max_y], gt_overlap_area)
        iou_scores.append(iou_score)

        if DEBUG:
            masked_image_copy = masked_image_org.copy()
            cv2.rectangle(masked_image_copy, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)
            cv2.imshow("masked_image_{}".format(intensity_threshold), masked_image_copy)
            cv2.imshow("masked_image", masked_image)
            cv2.waitKey(15)
    plt.plot(pixel_intensities, iou_scores)
    plt.show()
    print(iou_scores)
    print(pixel_intensities)
    return iou_scores, pixel_intensities


if __name__ == "__main__":
    DEBUG = False
    file_path = "D:\GitRepo\edge_computing\edge_collaboration\src"
    file_name = "view_1_area.JPG"

    image = cv2.imread("{}/{}".format(file_path, file_name))
    assert image is not None

    scores = weighted_spatial_overlap_performance(image, [28, 101, 617, 492])
    print(scores)
