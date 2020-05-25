import requests
import base64
import json
import numpy as np
import cv2


def detect_person(image_path):
    # image_path = "../examples/fish_bike.jpg"
    test_img = open(image_path, "rb")
    img_dict = {}  # dictionary containing image file

    img_dict["image"] = base64.b64encode(test_img.read()).decode('utf-8')

    # serialize and send to server
    json_obj = json.dumps(img_dict)
    server_url = "http://10.0.106.181:8080/mm/"
    # server_url = "http://10.4.20.6:5550/mm/"
    response = requests.post(server_url, json=json_obj)
    labels = response.content.decode("utf-8")
    # print("response : {}".format(labels))
    bboxes = json.loads(labels)
    print("bounding boxes: {}".format(bboxes))
    return bboxes
    # # analyse server response
    # input_image = cv2.imread(image_path)
    # input_image = cv2.resize(input_image, dsize=(300, 300))
    # classes = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
    #            'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant',
    #            'sheep', 'sofa', 'train', 'tvmonitor']
    #
    # for box in bboxes:
    #     classname = classes[int(box[0])]
    #     confidence = np.around(box[1] * 100, decimals=1)
    #     xmin = int(box[2])
    #     ymin = int(box[3])
    #     xmax = int(box[4])
    #     ymax = int(box[5])
    #
    #     # put rectangle on image
    #     obj = input_image[ymin:ymax, xmin:xmax]
    #     cv2.imshow(classname, obj)

    # cv2.waitKey(-1)
