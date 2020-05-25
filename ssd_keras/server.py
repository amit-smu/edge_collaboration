import numpy as np
import cv2
import socket
import tornado.ioloop
import tornado.web
import json
from tornado.options import define, options
from keras import backend as K
from keras.models import load_model
from keras.preprocessing import image
from keras.optimizers import Adam
from imageio import imread
import numpy as np
from matplotlib import pyplot as plt

from models.keras_ssd300 import ssd_300
from keras_loss_function.keras_ssd_loss import SSDLoss
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast
from keras_layers.keras_layer_L2Normalization import L2Normalization

from ssd_encoder_decoder.ssd_output_decoder import decode_detections, decode_detections_fast

from data_generator.object_detection_2d_data_generator import DataGenerator
from data_generator.object_detection_2d_photometric_ops import ConvertTo3Channels
from data_generator.object_detection_2d_geometric_ops import Resize
from data_generator.object_detection_2d_misc_utils import apply_inverse_transforms
import os
import cv2
import base64

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Set the image size.
img_height = 300
img_width = 300

# global variables
define("port", default=8080, type=int)
model = None


class MediaManager(tornado.web.RequestHandler):
    def get(self):
        print("get received..")

    def post(self, *args, **kwargs):
        global model
        print("post received..")
        request_body = self.request.body.decode('utf-8')
        json_obj = json.loads(request_body)
        json_obj = json.loads(json_obj)

        in_image = base64.b64decode(json_obj['image'])
        image_name = "temp_image.jpg"
        f = open(image_name, "wb")
        f.write(in_image)
        f.close()

        # pass this image to the object detector..
        print("doing object detection..")
        # read & resize input image
        orig_images = []  # Store the images here.
        input_images = []  # Store resized versions of the images here.

        # We'll only load one image in this example.
        img_path = 'temp_image.jpg'

        orig_images.append(imread(img_path))
        img = image.load_img(img_path, target_size=(img_height, img_width))
        img = image.img_to_array(img)
        input_images.append(img)
        input_images = np.array(input_images)
        print("shape of image : {}".format(input_images.shape))

        y_pred = model.predict(input_images)
        # print(y_pred)
        confidence_threshold = 0.5

        y_pred_thresh = [y_pred[k][y_pred[k, :, 1] > confidence_threshold] for k in range(y_pred.shape[0])]

        np.set_printoptions(precision=2, suppress=True, linewidth=90)
        print("Predicted boxes:\n")
        print('   class   conf xmin   ymin   xmax   ymax')
        print(y_pred_thresh[0])

        # send results back
        result = {'a': 1, 'b': 2}
        self.set_header("Content-Type", 'application/json')
        self.write(json.dumps(y_pred_thresh[0].tolist()))


if __name__ == '__main__':
    # IP_ADDRESS = socket.gethostbyname(socket.gethostname())
    IP_ADDRESS = "10.0.106.181"

    # initialize the DL model
    # model_path = './models/ssd300_model.h5'

    # # We need to create an SSDLoss object in order to pass that to the model loader.
    # ssd_loss = SSDLoss(neg_pos_ratio=3, n_neg_min=0, alpha=1.0)
    #
    # K.clear_session()  # Clear previous models from memory.
    #
    # model = load_model(model_path, custom_objects={'AnchorBoxes': AnchorBoxes,
    #                                                'L2Normalization': L2Normalization,
    #                                                'DecodeDetections': DecodeDetections,
    #                                                'compute_loss': ssd_loss.compute_loss})

    K.clear_session()  # Clear previous models from memory.
    model = ssd_300(image_size=(img_height, img_width, 3),
                    n_classes=20,
                    mode='inference',
                    l2_regularization=0.0005,
                    scales=[0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05],
                    # The scales for MS COCO are [0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05]
                    aspect_ratios_per_layer=[[1.0, 2.0, 0.5],
                                             [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                             [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                             [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                             [1.0, 2.0, 0.5],
                                             [1.0, 2.0, 0.5]],
                    two_boxes_for_ar1=True,
                    steps=[8, 16, 32, 64, 100, 300],
                    offsets=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                    clip_boxes=False,
                    variances=[0.1, 0.1, 0.2, 0.2],
                    normalize_coords=True,
                    subtract_mean=[123, 117, 104],
                    swap_channels=[2, 1, 0],
                    confidence_thresh=0.5,
                    iou_threshold=0.45,
                    top_k=200,
                    nms_max_output_size=400)

    weights_path = './trained_weights/VGG_VOC0712_SSD_300x300_iter_120000.h5'
    model.load_weights(weights_path, by_name=True)

    # 3: Compile the model so that Keras won't complain the next time you load it.
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)
    model.compile(optimizer=adam, loss=ssd_loss.compute_loss)

    assert model is not None
    print("model loaded..")

    # start the web server
    app = tornado.web.Application([
        (r'/mm/', MediaManager),
    ])
    app.listen(options.port, address=IP_ADDRESS)
    print("Server running at {}:{}...".format(IP_ADDRESS, options.port))
    tornado.ioloop.IOLoop.instance().start()
