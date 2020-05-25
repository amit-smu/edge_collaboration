import numpy as np
import cv2
import socket
import tornado.ioloop
import tornado.web
import json
import base64
import api

from tornado.options import define, options



# global variables
define("port", default=5550, type=int)


class MediaManager(tornado.web.RequestHandler):
    def get(self):
        print("get received..")

    def post(self, *args, **kwargs):
        print("post received..")
        request_body = self.request.body.decode('utf-8')
        json_obj = json.loads(request_body)
        json_obj = json.loads(json_obj)
        image = base64.b64decode(json_obj['image'])
        image_name = "temp_image.jpg"
        f = open(image_name, "wb")
        f.write(image)
        f.close()
        image = cv2.imread(image_name)
        assert image is not None
        # image = json_obj['image']

        # pass this image to the classifier..
        # print("THIS IS WHERE CLASSIFIER IS CALLED WITH IMAGE AS INPUT")

        img_embedding = api.human_vector(image)
        # print(type(img_embedding))
        # print("shape : {}".format(img_embedding.shape))
        img_embedding = img_embedding[0].tolist()
        # print(type(img_embedding))
        # print("shape : {}".format(len(img_embedding)))
        # print(img_embedding)

        # send results back
        result = {"embedding": img_embedding}
        self.set_header("Content-Type", 'application/json')
        result = json.dumps(result)
        # print("response : {}".format(result))
        self.write(result)


if __name__ == '__main__':
    # IP_ADDRESS = socket.gethostbyname(socket.gethostname())
    # IP_ADDRESS = "10.4.16.29"
    IP_ADDRESS = "172.17.0.7"
    app = tornado.web.Application([
        (r'/mm/', MediaManager),
    ])
    app.listen(options.port, address=IP_ADDRESS)
    print("Server running at {}:{}...".format(IP_ADDRESS, options.port))
    tornado.ioloop.IOLoop.instance().start()
