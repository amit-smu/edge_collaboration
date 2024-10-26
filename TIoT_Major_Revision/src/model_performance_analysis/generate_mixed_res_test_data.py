"""
module to generate mixed resolution dataset for testing the trained models
"""
import os
import glob
import cv2
import numpy as np


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


if __name__ == "__main__":
    # INPUT_DIR = "../dataset/trainval/test"
    INPUT_FILE = "test.txt"
    img_width, img_height = 1056, 1056
    np.random.seed(10)
    ICOV_THRESHOLD = 0.4

    TARGET_RES = 128
    TARGET_REGION = "33_middle"

    resolutions = [1056, 416, 224, 128]
    res_to_area = {
        "33_left": [0, 0, 349, 1056],
        "33_right": [697, 0, 1056, 1056],
        "33_middle": [349, 0, 697, 1056],
        "66_left": [0, 0, 697, 1056],
        "66_right": [349, 0, 1056, 1056],
        "66_middle":[180, 0, 876, 1056]
    }

    with open(INPUT_FILE, 'r') as test_list:
        filenames = [t.strip() for t in test_list.readlines()]
        for name in filenames:
            # name = name[3:]
            print(name, TARGET_RES, TARGET_REGION)
            image = cv2.imread(name)
            assert image is not None

            if TARGET_RES < 1056:
                sh_reg_res = TARGET_RES
                rand_overlap_reg = res_to_area[TARGET_REGION]
                region = image[rand_overlap_reg[1]: rand_overlap_reg[3],
                         rand_overlap_reg[0]: rand_overlap_reg[2]]
                region_h, region_w = region.shape[:2]
                region_h = float(region_h)
                region_w = float(region_w)
                region_w_tr = int((region_w / img_width) * sh_reg_res)
                region_h_tr = int((region_h / img_height) * sh_reg_res)
                temp = cv2.resize(region, dsize=(region_w_tr, region_h_tr), interpolation=cv2.INTER_AREA)
                region = cv2.resize(temp, dsize=(int(region_w), int(region_h)), interpolation=cv2.INTER_CUBIC)
                image[rand_overlap_reg[1]: rand_overlap_reg[3], rand_overlap_reg[0]: rand_overlap_reg[2]] = region
                # cv2.imwrite(name, image)

            prior = np.full(shape=(img_height, img_width, 3), fill_value=70, dtype=np.uint8)
            annot_file_path = f"{name[:-4]}.txt"
            with open(annot_file_path, 'r') as annot_file:
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
                        prior[ymin:ymax, xmin:xmax] = 250
                    annot = annot_file.readline().strip("\n").split(" ")
            # write prior to the file
            # cv2.imwrite(f"{name[:-4]}_prior.jpg", prior)
            demo_img = np.vstack((image, prior))
            cv2.imshow("final_image", demo_img)
            cv2.waitKey(-1)
