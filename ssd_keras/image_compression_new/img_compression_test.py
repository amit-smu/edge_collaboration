"""
testing image compression
"""
import numpy as np
import cv2
import time
from copy import deepcopy


def compress_PETS(input_file, sh_reg):
    img = cv2.imread(input_file)
    assert img is not None

    x1, y1, x2, y2 = sh_reg
    overlap_img = deepcopy(img[y1:y2, x1:x2])
    img[y1:y2, x1:x2] = 255
    cv2.imshow("temp", img)
    cv2.waitKey(-1)
    img = cv2.resize(img, dsize=(512, 512), interpolation=cv2.INTER_CUBIC)
    for q in JPEG_QUALITIES:
        iterations = 100
        start_time = time.time()
        for i in range(iterations):
            status, buffer = cv2.imencode(".jpg", img, params=[cv2.IMWRITE_JPEG_QUALITY, q])
        end_time = time.time()
        total_time = np.round(end_time - start_time, decimals=3)
        total_time = (1000 * total_time) / iterations
        print("Quality : {}, time_per_iter : {}, encoded_size: {}".format(q, total_time, len(buffer)))
        # find number of bytes in compressed image
        # temp = cv2.imread("temp/pets_{}.jpg".format(q))
        # print("Quality : {}, time : {}, size : {}".format(q, 1000 * (end_time - start_time), temp.shape))


def compress_WT(input_file, sh_reg):
    img = cv2.imread(input_file)
    assert img is not None

    x1, y1, x2, y2 = sh_reg
    overlap_img = deepcopy(img[y1:y2, x1:x2])
    img[y1:y2, x1:x2] = 255

    # resize image to 512x512 (DNN input image size)
    img = cv2.resize(img, dsize=(512, 512), interpolation=cv2.INTER_CUBIC)
    # cv2.imshow("temp", img)
    # cv2.waitKey(-1)
    for q in PNG_QUALITIES:
        iterations = 100
        start_time = time.time()
        for i in range(iterations):
            status, buffer = cv2.imencode(".png", img, params=[cv2.IMWRITE_PNG_COMPRESSION, q])
        # img = cv2.imdecode(b, cv2.IMREAD_COLOR)
        # img[y1:y2, x1:x2] = overlap_img
        # cv2.imwrite("temp/pets_{}.jpg".format(q), img)
        end_time = time.time()
        total_time = np.round(end_time - start_time, decimals=3)
        total_time = (1000 * total_time) / iterations
        print("Quality : {}, time_per_iter : {}, encoded_size: {}".format(q, total_time, len(buffer)))


if __name__ == "__main__":
    TEST_DATASET = "PETS"  # WT or PETS
    REF_CAM = 8
    COLLAB_CAM = 5

    JPEG_QUALITIES = [100, 95, 50, 25, 10, 5]
    PNG_QUALITIES = [1, 3, 5, 7, 8, 9]

    shared_reg_key = "{}_{}_{}".format(TEST_DATASET, REF_CAM, COLLAB_CAM)

    shared_regions = {
        "PETS_5_7": [91, 142, 693, 510],
        "PETS_8_5": [267, 85, 716, 537],
        "WT_1_4": [6, 4, 908, 1074],
        "WT_5_7": [51, 139, 1507, 1041]
    }

    if TEST_DATASET == "WT":
        in_file = "D:\GitRepo\edge_computing\edge_collaboration\dataset\Wildtrack_dataset\PNGImages\C{}_00000000.png".format(
            REF_CAM)
    elif TEST_DATASET == "PETS":
        in_file = "D:\GitRepo\edge_computing\edge_collaboration\dataset\PETS_org\JPEGImages/frame_{}_0062.jpg".format(
            REF_CAM)

    sh_reg = shared_regions[shared_reg_key]
    print("Shared_reg : {}".format(sh_reg))

    if TEST_DATASET == "PETS":
        compress_PETS(in_file, sh_reg)
    elif TEST_DATASET == "WT":
        compress_WT(in_file, sh_reg)
