"""
script used for idealized collaboration performance modelling -- used to extract actual shared region out of an image
"""

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
    in_data_dir = r"../../../../rpi_hardware/raw_image_processing/data/episode_1/pi_2_frames_1056_v2"
    label_dir = r"../../../../rpi_hardware/raw_image_processing/data/episode_1/ground_truth/frame_wise_gt_yolo/cam_2"
    frame_numbers_path = "./test_frame_numbers.txt"
    out_data_dir = "./data"
    cam_pair = "2_1"
    ref_cam = 2
    IOU_TH = 0.2

    resolutions = [1056, 512, 416, 320, 224, 128, 70]
    shared_region_res = 1056

    img_width, img_height = (1056.0, 1056.0)
    spatial_overlap = {
        "2_1": [360, 0, 1010, 1080],
        "2_3": [0, 0, 470, 1080],
        "1_2": [0, 0, 480, 1080],
        "3_2": []
    }
    scale = 1056 / 1080
    spatial_overlap[cam_pair] = [int(d * scale) for d in spatial_overlap[cam_pair]]

    x1, y1, x2, y2 = spatial_overlap[cam_pair]
    print("Cam pair: {}, Spatial Overlap : {}, Resolution: {}\n".format(cam_pair, spatial_overlap[cam_pair],
                                                                        shared_region_res))
    sh_width = x2 - x1
    sh_height = y2 - y1

    with open(frame_numbers_path) as in_file:
        frame_numbers = in_file.read().split("\n")
    for number in frame_numbers:
        if len(number) == 0:
            continue
        frame_name = "frame_{}_{}.jpg".format(ref_cam, number)
        print(frame_name)
        image = cv2.imread("{}/{}".format(in_data_dir, frame_name))
        assert image is not None

        # extract and downsample the shared region
        sh_region = image[y1:y2, x1:x2]
        sh_width_new = int((sh_width / img_width) * shared_region_res)
        sh_height_new = int((sh_height / img_height) * shared_region_res)
        temp = cv2.resize(sh_region, dsize=(sh_width_new, sh_height_new), interpolation=cv2.INTER_AREA)
        sh_region = cv2.resize(temp, dsize=(sh_width, sh_height), interpolation=cv2.INTER_CUBIC)
        # image[:, :, :] = 0
        # image[y1:y2, x1:x2] = sh_region
        # sh_region = image[y1 - 1:y2 + 1, x1 - 1:x2 + 1]

        # cv2.imshow("new_img", image)
        # cv2.waitKey(-1)
        # add border to image (like a padding)
        image = cv2.copyMakeBorder(sh_region, y1, int(img_height - y2), x1, int(img_width - x2),
                                   borderType=cv2.BORDER_CONSTANT,
                                   value=0)
        cv2.imwrite("{}/{}".format(out_data_dir, frame_name), image)

        label_name = "{}.txt".format(frame_name[:-4])
        # print(label_name)
        prior = np.full(shape=(int(img_height), int(img_width), 1), fill_value=0, dtype=np.uint8)
        with open("{}/{}".format(label_dir, label_name)) as label_file:
            content = label_file.readlines()

            for line in content:
                if len(line) <= 1:
                    continue
                line = line.strip()

                # read gt for this image
                _, mid_x, mid_y, box_width, box_height = [x for x in line.split()]

                mid_x = float(mid_x) * img_width
                mid_y = float(mid_y) * img_height
                box_width = float(box_width) * img_width
                box_height = float(box_height) * img_height

                # enforce constraints
                left = int(max(1, mid_x - box_width / 2))
                top = int(max(1, mid_y - box_height / 2))
                right = int(min(mid_x + box_width / 2, img_width - 1))
                bottom = int(min(mid_y + box_height / 2, img_height - 1))

                #  check overlap with shared region
                if bb_icov(gt_box=[left, top, right, bottom], cropped_img_box=spatial_overlap[cam_pair]) >= IOU_TH:
                    prior[top:bottom, left:right] = 255
        prior_subset = prior[y1:y2, x1:x2]
        cv2.imwrite("{}/{}_prior.jpg".format(out_data_dir, label_name[:-4]), prior)