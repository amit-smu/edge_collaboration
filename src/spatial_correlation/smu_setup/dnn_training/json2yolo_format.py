"""
convert json file (ground truth manually annotated) to yolo format
"""

import json
import os
import numpy as np
import cv2


def get_key(x):
    x = x[:-5].split("_")[1]
    return int(x)


if __name__ == "__main__":
    IN_DIR = r"./data/episode_1/ground_truth/frame_wise_gt_json/"
    OUT_DIR = r"./data/episode_1/ground_truth/frame_wise_gt_yolo/"
    IMAGE_SIZE = 1056
    TEST_LABELS = False

    for cam in range(1, 4):
        cam_dir = "{}/cam_{}".format(IN_DIR, cam)
        cam_out_dir = "{}/cam_{}".format(OUT_DIR, cam)
        filenames = os.listdir(cam_dir)
        filenames = sorted(filenames, key=lambda x: get_key(x))

        for name in filenames:
            print(name)
            f_num = name[:-5].split("_")[2]
            yolo_filename = "{}/frame_{}_{}.txt".format(cam_out_dir, cam, f_num)
            with open("{}/{}".format(cam_dir, name)) as in_file:
                j_obj = json.load(in_file)
                with open(yolo_filename, 'w') as yolo_out_file:
                    for obj in j_obj:
                        coords = obj["coords"]
                        coords = [int(i) for i in coords]
                        coords = np.clip(coords, 0, IMAGE_SIZE)
                        x, y, w, h = coords
                        x_mid = (x + w / 2) / IMAGE_SIZE
                        y_mid = (y + h / 2) / IMAGE_SIZE
                        w = w / IMAGE_SIZE
                        h = h / IMAGE_SIZE
                        yolo_out_file.write("{} {} {} {} {}\n".format(0, x_mid, y_mid, w, h))

            if TEST_LABELS:
                img_dir = r"./data/episode_1/pi_{}_frames_1056_v2".format(cam)
                img_name = "frame_{}_{}.jpg".format(cam, f_num)
                img = cv2.imread("{}/{}".format(img_dir, img_name))
                assert img is not None
                with open("{}/frame_{}_{}.txt".format(cam_out_dir, cam, f_num)) as input_file:
                    labels = input_file.read().strip().split("\n")
                    for label in labels:
                        label = label.split(" ")[1:]
                        label = [float(x) for x in label]
                        mid_x, mid_y, width, height = [int(x * IMAGE_SIZE) for x in label]
                        x1 = int(mid_x - width / 2)
                        y1 = int(mid_y - height / 2)
                        x2 = int(mid_x + width / 2)
                        y2 = int(mid_y + height / 2)
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.imshow("img", img)
                cv2.waitKey(-1)
