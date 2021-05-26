"""
module to create training data for image + prior model.
Images would contain randomly generated regions of various resolutions and correpsonding ground truth as fourth channel
"""

import cv2
import random
import numpy as np
from shutil import copyfile

np.random.seed(0)
random.seed(0)


def bb_icov(gt_box, cropped_img_box):
    # determine the (x, y)-coordinates of the intersection rectangle
    boxA = gt_box
    boxB = cropped_img_box
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    # boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea)
    return np.round(iou, decimals=2)


def get_rand_overlap_region():
    resolutions = [720, 512, 360, 224, 160, 96]
    shared_reg_res = np.random.choice(resolutions)
    # select (x1, y1, x2, y2 in percentage)
    x1_rand = np.random.randint(0, 40)
    y1_rand = np.random.randint(0, 40)
    x2_rand = np.random.randint(60, 80)
    y2_rand = np.random.randint(60, 80)
    # translate as per image size
    x1 = int((x1_rand * img_width) / 100)
    y1 = int((y1_rand * img_height) / 100)
    x2 = int((x2_rand * img_width) / 100)
    y2 = int((y2_rand * img_height) / 100)
    return shared_reg_res, [x1, y1, x2, y2]


def randomize_prior(prior, obj_coords, randomize):
    r_int_1 = np.random.randint(1, 10)
    if r_int_1 > 8:
        return prior
    xmin, ymin, xmax, ymax = obj_coords
    if randomize:
        # r_int = np.random.randint(1, 10)
        # if r_int <= 8:
        # if True:
        width_rnum = np.random.uniform(-0.15, 0.15)
        height_rnum = np.random.uniform(-0.15, 0.15)
        obj_width = xmax - xmin
        obj_height = ymax - ymin

        xmin = int(xmin + (obj_width * width_rnum))
        ymin = int(ymin + (obj_height * height_rnum))
        xmax = int(xmin + obj_width + (obj_width * width_rnum))
        ymax = int(ymin + obj_height + (obj_height * height_rnum))
        # check for out of frame values
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(img_width, xmax)
        ymax = min(img_height, ymax)

        prior[ymin:ymax, xmin:xmax] = 255
        # else:
        #     rand_int = random.randint(1, 2)
        #     if rand_int == 1:
        #         prior[:, :] = 114
        #     else:
        #         prior[:, :] = 255
    else:
        prior[ymin:ymax, xmin:xmax] = 255
    return prior


if __name__ == "__main__":
    img_dir = "dataset/Wildtrack_dataset/PNGImages/"
    annot_dir = "dataset/Wildtrack_dataset/PNGImages/"
    img_output_dir = "dataset/Wildtrack_dataset_transp_prior/PNGImages/"
    training_files = "all_filenames.txt"
    img_height, img_width = (1056, 1056)
    ICOV_THRESHOLD = 0.5

    with open(training_files) as train_files:
        filename = train_files.readline().strip("\n").split("/")[-1]
        while len(filename) > 2:
            print(filename)
            image = cv2.imread("{}/{}".format(img_dir, filename))
            assert image is not None
            # prior = np.ndarray(shape=(img_height, img_width), dtype=np.uint8)
            prior = np.full(shape=(img_height, img_width), fill_value=255, dtype=np.uint8)

            final_img = np.ndarray(shape=(img_height, img_width, 4), dtype=np.uint8)
            final_img[:, :, :3] = image
            final_img[:, :, 3] = prior
            print("img shape : {}".format(final_img.shape))

            annot_file_path = "{}/{}.txt".format(img_dir, filename[:-4])
            cv2.imwrite("{}/{}".format(img_output_dir, filename), final_img)
            copyfile(src=annot_file_path, dst="{}/{}.txt".format(img_output_dir, filename[:-4]))

            filename = train_files.readline().strip("\n").split("/")[-1]
            # break
