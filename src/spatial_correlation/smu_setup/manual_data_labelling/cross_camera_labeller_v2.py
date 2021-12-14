"""
module to help with labelling of cross-camera object tracking
"""

import cv2
import copy
import os, sys
import numpy as np
import numpy as np
from PIL import Image


def get_key(x):
    # print(x)
    x = x.split("_")[-1]
    return int(x)


def load_detected_objects(filename):
    """
    use the detected objects data (from darknet) and convert it to a python dict
    :return: """
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
    global prev_obj, first_time
    color_index = 0
    cv2.putText(img, "Frame_{}".format(frame_num), (100, 100), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)
    for obj in objects:
        # re-scale objects to 1056x1056 (detections were made at 1080x1080)
        ratio = 1056 / 1080
        obj = [int(x * ratio) for x in obj]

        rand = np.random.randint(0, len(COLORS))
        color = COLORS[color_index]
        color_index = (color_index + 1) % len(COLORS)
        # cv2.rectangle(img, (obj[0], obj[1]), (obj[0] + obj[2], obj[1] + obj[3]), color, 1)

        # cv2.putText(img, "{},{},{},{}".format(obj[0], obj[1], obj[2], obj[3]), (obj[0], obj[1]), cv2.FONT_HERSHEY_PLAIN,
        #            2, color, 2)
    return img


if __name__ == "__main__":
    os.system("cp input_new.log input.log")
    uid = []
    os.system("rm -rf pi*_data/*")
    json = open("ep_1_gt_frames_1056.json", "w")
    json.write('{\n')
    json.close()
    detail = [{}, {}, {}]
    prev_detail = [{}, {}, {}]
    prev_index = [0, 0, 0]
    prev_nums = [0] * 400
    user_detail = {}
    IMAGE_DIMS = (1056, 1056)
    COLORS = [(0, 255, 0), (255, 255, 0), (0, 0, 255), (255, 0, 255,), (255, 0, 0), (15, 250, 150), (190, 0, 100)]
    frame_index = 1
    frames_dir = r"./episode_1/pi_{}_frames_1056"
    cur_box_uid = 0
    # pi_2_dir = ""
    # pi_3_dir = ""

    images_dir = "./output_images_with_label"
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
    os.system("rm -rf " + images_dir + "/*")
    pi_1_detections = r"./episode_1/result_pi_1_single_detections.txt"
    pi_2_detections = r"./episode_1/result_pi_2_single_detections.txt"
    pi_3_detections = r"./episode_1/result_pi_3_single_detections.txt"

    pi_1_objects = load_detected_objects(pi_1_detections)
    pi_2_objects = load_detected_objects(pi_2_detections)
    pi_3_objects = load_detected_objects(pi_3_detections)

    pi_1_frame_names = sorted(list(pi_1_objects.keys()), key=lambda x: get_key(x))

    frame_counter = 0
    for name in pi_1_frame_names:
        if frame_counter > 0:
            frame_counter = (frame_counter + 1) % 8
            continue
        if frame_index < 433:
            frame_index += 1
            frame_counter = (frame_counter + 1) % 8
            continue
        f_num = "{}".format(get_key(name))
        print("\n************Frame Number - {} **********************".format(f_num))
        # read images from all pi
        print("Total frames analysed", frame_index)

        print("{}/frame_1_{}.jpg".format(frames_dir.format(1), f_num))
        pi_1_img = cv2.imread("{}/frame_1_{}.jpg".format(frames_dir.format(1), f_num))
        pi_2_img = cv2.imread("{}/frame_2_{}.jpg".format(frames_dir.format(2), f_num))
        pi_3_img = cv2.imread("{}/frame_3_{}.jpg".format(frames_dir.format(3), f_num))
        assert pi_1_img is not None
        assert pi_2_img is not None
        assert pi_3_img is not None

        pi_1_obj = pi_1_objects["frame_1_{}".format(f_num)]
        pi_2_obj = pi_2_objects["frame_2_{}".format(f_num)]
        pi_3_obj = pi_3_objects["frame_3_{}".format(f_num)]

        # draw objects on imags
        pi_1_img = draw_objects(pi_1_img, pi_1_obj, 1, f_num)
        pi_2_img = draw_objects(pi_2_img, pi_2_obj, 2, f_num)
        pi_3_img = draw_objects(pi_3_img, pi_3_obj, 3, f_num)

        pi_obj = [pi_1_obj, pi_2_obj, pi_3_obj]
        pi_img = [pi_1_img, pi_2_img, pi_3_img]
        pi_img2 = [pi_1_img.copy(), pi_2_img.copy(), pi_3_img.copy()]
        i = 0
        id_no = 434

        k = 7
        json = open("ep_1_gt_frames_1056.json", "a")
        json.write('    "' + str(f_num) + '": {\n')
        json.write('        "frame_num": ' + str(f_num) + ',\n')
        # json.write('}\n')
        for i in range(3):
            for obj in pi_obj[i]:
                for k in range(3):
                    if (prev_index[k] == 1):
                        prev_index[k] = 0
                        prev_detail[k] = {}
                        prev_detail[k] = copy.deepcopy(detail[k])
                        detail[k] = {}
                ratio = 1056 / 1080
                obj = [int(x * ratio) for x in obj]
                cords = obj
                y = int(cords[1])
                if y < 0:
                    y = 0
                y_end = y + int(cords[3])
                if (y_end > 1055):
                    y_end = 1055
                x = int(cords[0])
                x_end = x + int(cords[2])
                for k in range(3):
                    for key in prev_detail[k]:
                        cv2.putText(pi_img[k], key, (prev_detail[k][key][2] - 10, prev_detail[k][key][3] - 20),
                                    cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
                    for key in detail[k]:
                        cv2.putText(pi_img[k], key, (detail[k][key][0] + 10, detail[k][key][1] + 20),
                                    cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
                if x < 0:
                    x = 0
                if (x_end > 1055):
                    x_end = 1055
                cropped = pi_img[i][y:y_end, x:x_end]
                pi_img_3 = pi_img[i].copy()
                cv2.rectangle(pi_img_3, (x, y), (x_end, y_end), (255, 0, 0), 3)
                result = [pi_img[0], pi_img[1], pi_img[2]]
                result[i] = pi_img_3
                im_v = cv2.hconcat(result)
                cv2.imwrite("curr.png", im_v)
                cropped_resize = cv2.resize(cropped, ((x_end - x) * 3, (y_end - y) * 3), cv2.INTER_CUBIC)
                cv2.imwrite("curr_user.png", cropped_resize)
                avail = max(uid, default=-1)
                while (1):
                    try:
                        user_id = input("enter user id for " + str(obj) + " in pi " + str(
                            i + 1) + " enter 0 to ignor used upto " + str(avail) + ": ")
                        uid.append(int(user_id))
                        if ((user_id in detail[i]) and (int(user_id) != 0)):
                            print("\ntry again with different id\n")
                        else:
                            with open("input_new.log", "a") as log_file:
                                log_file.write(str(cur_box_uid) + " " + user_id + " " + str(f_num) + "\n")
                            break;
                    except KeyboardInterrupt:
                        os._exit(0)
                    except:
                        if (user_id == ''):
                            with open("input.log", "r") as log_file:
                                vals = log_file.read()
                                vals = vals.splitlines()
                                hit = 0
                                for val in vals:
                                    xx = val.split()
                                    curr = int(xx[0])
                                    if (curr == cur_box_uid):
                                        user_id = xx[1]
                                        print("taking from past inputs ", xx)
                                        with open("input_new.log", "a") as log_file2:
                                            log_file2.write(val)
                                        uid.append(int(user_id))
                                        hit = 1
                                if (hit):
                                    break

                        pass
                cur_box_uid += 1
                detail[i][user_id] = [x, y, x_end, y_end, obj]

                directory = "./pi" + str(i + 1) + "_data/user_" + user_id
                if not os.path.exists(directory):
                    os.makedirs(directory)
                cv2.imwrite(directory + "/frame_" + str(f_num) + ".jpg", cropped)
                with open(directory + "/pos.txt", "a") as writter:
                    writter.write(str(f_num) + " " + ",".join(list(map(str, obj))) + "\n")
        for k in range(3):
            for key in detail[k]:
                if (int(key) != 0):
                    cv2.putText(pi_img2[k], key, (detail[k][key][0] + 10, detail[k][key][1] + 30),
                                cv2.FONT_HERSHEY_PLAIN, 2, COLORS[int(key) % 7], 2)
                    cv2.rectangle(pi_img2[k], (detail[k][key][0], detail[k][key][1]),
                                  (detail[k][key][2], detail[k][key][3]), COLORS[int(key) % 7], 2)
        result = [pi_img2[0], pi_img2[1], pi_img2[2]]
        im_v = cv2.hconcat(result)
        cv2.imwrite(images_dir + "/curr" + f_num + ".png", im_v)
        for i in range(3):
            json.write('        "cam_' + str(i + 1) + '": [\n')
            current_key = 1
            first_entry = 1
            for key in detail[i]:
                if (int(key) != 0):
                    if (first_entry == 0):
                        json.write(',\n')
                    else:
                        first_entry = 0
                    json.write('            {\n')
                    json.write('                "obj_id": ' + str(int(key) + 200) + ',\n')
                    json.write('                "coords" :  ' + str(detail[i][key][-1]) + '\n')
                    json.write('            }')
            json.write('\n')
            json.write('        ]')
            if (i < 2):
                json.write(',\n')
            else:
                json.write('\n    }')
        json.write(',\n')
        json.close()
        frame_counter = (frame_counter + 1) % 8
        prev_index[0] = 1
        prev_index[1] = 1
        prev_index[2] = 1
        frame_index += 1

        # cv2.destroyAllWindows()
cross_camera_data_labeller.py
Displaying
cross_camera_data_labeller.py.