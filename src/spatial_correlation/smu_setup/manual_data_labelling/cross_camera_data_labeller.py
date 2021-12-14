"""
module to help with labelling of cross-camera object tracking
"""

import cv2
import os, sys
import numpy as np
import json


def get_key(x):
    # print(x)
    x = x.split("_")[-1]
    return int(x)


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


def draw_objects(img, objects, pi_number, frame_num):
    color_index = 0
    print("\nPi_{}..".format(pi_number))
    cv2.putText(img, "Frame_{}".format(frame_num), (100, 100), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
    for obj in objects:
        # re-scale objects to 1056x1056 (detections were made at 1080x1080)
        ratio = 1056 / 1080
        obj = [int(x * ratio) for x in obj]

        print("{},{},{},{}".format(obj[0], obj[1], obj[2], obj[3]))
        rand = np.random.randint(0, len(COLORS))
        color = COLORS[color_index]
        color_index = (color_index + 1) % len(COLORS)
        cv2.rectangle(img, (obj[0], obj[1]), (obj[0] + obj[2], obj[1] + obj[3]), color, 1)

        cv2.putText(img, "{},{},{},{}".format(obj[0], obj[1], obj[2], obj[3]), (obj[0], obj[1]), cv2.FONT_HERSHEY_PLAIN,
                    2, color, 2)
    return img


def draw_objects_interactive(img, obj_list, cam_num):
    """
    draw objects one by one and prompt for their unique ID
    :param img:
    :param obj_list:
    :return:
    """
    json_obj_list = []

    obj_list = sorted(obj_list, key=lambda x: int(x[0]))

    for obj in obj_list:
        # re-scale objects to 1056x1056 (detections were made at 1080x1080)
        ratio = 1056 / 1080
        obj = [int(x * ratio) for x in obj]

        img_copy = img.copy()
        obj = np.clip(obj, 0, 1055)
        cv2.rectangle(img_copy, (obj[0], obj[1]), (obj[0] + obj[2], obj[1] + obj[3]), (0, 255, 0), 1)
        cv2.putText(img_copy, "{},{},{},{}".format(obj[0], obj[1], obj[2], obj[3]), (obj[0], obj[1]),
                    cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        cv2.imshow("pi_cam_{}".format(cam_num), img_copy)
        cv2.waitKey(300)
        # clip object

        # prompt for user id
        user_id = input("Enter User Id: ")
        if not user_id.isdigit():
            continue
        # cv2.destroyAllWindows()
        print(user_id)
        j = {
            "obj_id": int(user_id),
            "coords": obj.tolist()
        }
        json_obj_list.append(j)

    return json_obj_list


if __name__ == "__main__":
    IMAGE_DIMS = (1056, 1056)
    COLORS = [(0, 255, 0), (255, 255, 0), (0, 0, 255), (255, 0, 255,), (255, 0, 0), (15, 250, 150), (190, 0, 100)]
    frame_index = 1

    frames_dir = r"../../rpi_hardware/raw_image_processing/data/episode_1/pi_{}_frames_1056"
    # pi_2_dir = ""
    # pi_3_dir = ""

    gt_json_dir = r"../../rpi_hardware/raw_image_processing/data/episode_1/ground_truth/gt_json"
    pi_1_detections = r"../../rpi_hardware/raw_image_processing/data/episode_1/result_pi_1_single_detections.txt"
    pi_2_detections = r"../../rpi_hardware/raw_image_processing/data/episode_1/result_pi_2_single_detections.txt"
    pi_3_detections = r"../../rpi_hardware/raw_image_processing/data/episode_1/result_pi_3_single_detections.txt"

    pi_1_objects = load_detected_objects(pi_1_detections)
    pi_2_objects = load_detected_objects(pi_2_detections)
    pi_3_objects = load_detected_objects(pi_3_detections)

    pi_1_frame_names = sorted(list(pi_1_objects.keys()), key=lambda x: get_key(x))

    frame_counter = 0
    for name in pi_1_frame_names:
        if frame_counter > 0:
            frame_counter = (frame_counter + 1) % 8
            continue

        f_num = "{}".format(get_key(name))

        if int(f_num) < 4864:
            continue

        print("\n************Frame Number - {} **********************".format(f_num))
        print("Total Frames Analyzed: {}".format(frame_index))
        # read images from all pi
        pi_1_img = cv2.imread("{}/frame_1_{}.jpg".format(frames_dir.format(1), f_num))
        pi_2_img = cv2.imread("{}/frame_2_{}.jpg".format(frames_dir.format(2), f_num))
        pi_3_img = cv2.imread("{}/frame_3_{}.jpg".format(frames_dir.format(3), f_num))
        assert pi_1_img is not None
        assert pi_2_img is not None
        assert pi_3_img is not None

        pi_1_obj = pi_1_objects["frame_1_{}".format(f_num)]
        pi_2_obj = pi_2_objects["frame_2_{}".format(f_num)]
        pi_3_obj = pi_3_objects["frame_3_{}".format(f_num)]

        # draw objects one by one
        # create a json object
        dict_key = "{}".format(f_num)
        gt_dict = {
            dict_key: {}
        }
        gt_dict[dict_key]["frame_num"] = int(f_num)

        pi_images = [pi_1_img, pi_2_img, pi_3_img]
        pi_detections = [pi_1_obj, pi_2_obj, pi_3_obj]
        for cam_num in range(1, 4):
            print("Camera Number : {}".format(cam_num))
            obj_list = draw_objects_interactive(pi_images[cam_num - 1], pi_detections[cam_num - 1], cam_num)
            # print(obj_list)
            gt_dict[dict_key]["cam_{}".format(cam_num)] = obj_list

        gt_filename = "frame_{}.json".format(f_num)
        with open("{}/{}".format(gt_json_dir, gt_filename), 'w') as out_file:
            gt_json = json.dump(gt_dict, out_file, indent=4, sort_keys=True)

        # put json object to file
        # out_json_file.write(j_obj)
        # out_json_file.flush()

        # print(j_obj)
        cv2.destroyAllWindows()
        # draw objects on imags
        # pi_1_img = draw_objects(pi_1_img, pi_1_obj, 1, f_num)
        # pi_2_img = draw_objects(pi_2_img, pi_2_obj, 2, f_num)
        # pi_3_img = draw_objects(pi_3_img, pi_3_obj, 3, f_num)
        #
        # # show images
        #
        # pi_1_img = cv2.resize(pi_1_img, (500, 500))
        # pi_2_img = cv2.resize(pi_2_img, (500, 500))
        # pi_3_img = cv2.resize(pi_3_img, (500, 500))
        # cv2.imshow("Pi_1", pi_1_img)
        # cv2.imshow("Pi_2", pi_2_img)
        # cv2.imshow("Pi_3", pi_3_img)
        #
        # cv2.waitKey(-1)
        frame_counter = (frame_counter + 1) % 8
        frame_index += 1
        # cv2.destroyAllWindows()
