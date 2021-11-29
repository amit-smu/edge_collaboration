import cv2





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
                draw_boxes(key, obj_list)
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


load_detected_objects("result_4.txt")
