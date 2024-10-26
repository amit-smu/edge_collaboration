"""
module to generate priors for the training and test datasets
Logic is inherited from previous trainings
"""

import os

import cv2
import numpy as np


def get_rand_overlap_region():
    resolutions = [1056, 512, 416, 320, 224, 128]
    shared_reg_res = np.random.choice(resolutions)
    # shared_reg_res = 128
    # select (x1, y1, x2, y2 in percentage)
    x1_rand = np.random.randint(0, 40)
    y1_rand = np.random.randint(0, 40)
    x2_rand = np.random.randint(70, 90)
    y2_rand = np.random.randint(70, 90)
    # translate as per image size
    x1 = int((x1_rand * img_width) / 100)
    y1 = int((y1_rand * img_height) / 100)
    x2 = int((x2_rand * img_width) / 100)
    y2 = int((y2_rand * img_height) / 100)
    return shared_reg_res, [x1, y1, x2, y2]


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


def randomize_prior(prior, obj_coords, randomize):
    r_int_1 = np.random.randint(1, 10)
    if r_int_1 > 8:
        return prior
    xmin, ymin, xmax, ymax = obj_coords
    if randomize:
        width_rnum = np.random.uniform(-0.05, 0.05)
        height_rnum = np.random.uniform(-0.05, 0.05)
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

    prior[ymin:ymax, xmin:xmax] = 250
    return prior


if __name__ == "__main__":
    INPUT_DIRS = "../dataset/trainval/train"
    img_width, img_height = 1056, 1056
    np.random.seed(10)
    ICOV_THRESHOLD = 0.4

    # generate priors for training dataset
    img_list = os.listdir(f"{INPUT_DIRS}")
    for img in img_list:
        if not (img.__contains__("jpg")):
            continue
        print(img)
        image = cv2.imread(f"{INPUT_DIRS}/{img}")
        assert image is not None

        sh_reg_res, rand_overlap_reg = get_rand_overlap_region()
        region = image[rand_overlap_reg[1]: rand_overlap_reg[3], rand_overlap_reg[0]: rand_overlap_reg[2]]
        region_h, region_w = region.shape[:2]
        region_h = float(region_h)
        region_w = float(region_w)
        region_w_tr = int((region_w / img_width) * sh_reg_res)
        region_h_tr = int((region_h / img_height) * sh_reg_res)
        temp = cv2.resize(region, dsize=(region_w_tr, region_h_tr), interpolation=cv2.INTER_AREA)
        region = cv2.resize(temp, dsize=(int(region_w), int(region_h)), interpolation=cv2.INTER_CUBIC)
        image[rand_overlap_reg[1]: rand_overlap_reg[3], rand_overlap_reg[0]: rand_overlap_reg[2]] = region
        # cv2.imwrite(f"{INPUT_DIRS}/{img}", image)

        # load annotation file
        prior = np.full(shape=(img_height, img_width, 1), fill_value=70, dtype=np.uint8)
        # prior = image
        annot_file_path = f"{INPUT_DIRS}/{img[:-4]}.txt"

        r_int = np.random.randint(1, 10)
        if r_int > 9:
            # write empty prior to a file
            # demo_img = np.vstack((image, prior))
            cv2.imwrite(f"{INPUT_DIRS}/{img[:-4]}_prior.jpg", prior)
            continue
        with open(annot_file_path) as annot_file:
            annot = annot_file.readline().strip("\n").split(" ")
            while len(annot) > 1:
                mid_x = float(annot[1]) * img_width
                mid_y = float(annot[2]) * img_height
                width = float(annot[3]) * img_width
                height = float(annot[4]) * img_height

                xmin = int(mid_x - width / 2)
                xmax = int(mid_x + width / 2)
                ymin = int(mid_y - height / 2)
                ymax = int(mid_y + height / 2)

                icov_score = bb_icov(gt_box=[xmin, ymin, xmax, ymax], cropped_img_box=rand_overlap_reg)
                if icov_score >= ICOV_THRESHOLD:
                    prior = randomize_prior(prior, [xmin, ymin, xmax, ymax], randomize=True)
                annot = annot_file.readline().strip("\n").split(" ")
        # write prior to the file
        cv2.imwrite(f"{INPUT_DIRS}/{img[:-4]}_prior.jpg", prior)
        # break
        # demo_img = np.vstack((image, prior))
        # cv2.imshow("final_image", demo_img)
        # cv2.waitKey(-1)
