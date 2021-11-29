"""
module to help with labelling of cross-camera object tracking
"""

import cv2
import os, sys


def load_detected_objects(filename):
    """
    use the detected objects data (from darknet) and convert it to a python dict
    :return:
    """
    detected_objs = {}
    with open(filename) as in_file:
        line = in_file.readline()
        key = line.split("/")[-1].split(":")[0][:-4]
        obj_list = []
        while len(line) > 5:
            line = in_file.readline()
            if line.__contains__("Enter"):
                detected_objs[key] = obj_list
                # draw_boxes(key, obj_list)
                obj_list = []
                key = line.split("/")[-1].split(":")[0][:-4]
                # print(detected_objs)
            else:
                obj_class = line.split(":")[0]
                if obj_class != "person":
                    continue
                x, y, w, h = line.strip().split(", ")[1].split(" ")
                x = int(x)
                y = int(y)
                w = int(w)
                h = int(h)
                obj_list.append([x, y, w, h])
    return detected_objs


if __name__ == "__main__":
    IMAGE_DIMS = (1056, 1056)
    COLORS = [(0, 255, 0), (255, 255, 0), (0, 0, 255), (255, 0, 255,)]

    frames_dir = r"../../rpi_hardware/raw_image_processing/data/episode_1/pi_{}_frames"
    # pi_2_dir = ""
    # pi_3_dir = ""

    pi_1_detections = ""
    pi_2_detections = ""
    pi_3_detections = ""
