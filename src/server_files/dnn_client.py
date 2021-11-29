"""
module to send requests to dnn server for receiving embeddings (128-D vector) of a given image file.
"""

import requests
import base64
import json

# server_url = "http://10.4.16.29:5550/mm/"
server_url = "http://10.0.109.82:5917/mm/"
# server_url = "http://172.17.0.5:5950/mm/"


def get_embeddings(test_img_path):
    # open image in binary mode
    test_img = open("{}".format(test_img_path), "rb")
    img_dict = {}  # dictionary containing image file
    img_dict["image"] = base64.b64encode(test_img.read()).decode('utf-8')

    # serialize and send to server
    json_obj = json.dumps(img_dict)

    response = requests.post(server_url, json=json_obj)
    # print(type(response))
    # print(response)
    response_str = response.content.decode("utf-8")
    response_json = json.loads(response_str)
    embedding = response_json['embedding']
    # print(embedding)
    # print("response : {}".format(labels))

    return embedding

# print(get_embeddings(r"C:\Users\tango\Downloads\\color_corrected2.jpg"))