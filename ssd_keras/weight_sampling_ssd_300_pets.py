import h5py
import numpy as np
import shutil
from misc_utils.tensor_sampling_utils import sample_tensors
import cv2
from matplotlib import pyplot as plt
from imageio import imread

from keras.optimizers import Adam
from keras import backend as K
from keras.models import load_model
from keras.preprocessing import image

from models.keras_ssd300 import ssd_300
from keras_loss_function.keras_ssd_loss import SSDLoss
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast
from keras_layers.keras_layer_L2Normalization import L2Normalization

from data_generator.object_detection_2d_data_generator import DataGenerator
from data_generator.object_detection_2d_photometric_ops import ConvertTo3Channels
from data_generator.object_detection_2d_patch_sampling_ops import RandomMaxCropFixedAR
from data_generator.object_detection_2d_geometric_ops import Resize

if __name__ == "__main__":

    # TODO: Set the path for the source weights file you want to load.

    weights_source_path = './trained_weights/VGG_coco_SSD_300x300_iter_400000.h5'

    # TODO: Set the path and name for the destination weights file
    #       that you want to create.

    weights_destination_path = './trained_weights/VGG_coco_SSD_300x300_iter_400000_subsampled_8_classes.h5'

    # Make a copy of the weights file.
    shutil.copy(weights_source_path, weights_destination_path)

    # Load both the source weights file and the copy we made.
    # We will load the original weights file in read-only mode so that we can't mess up anything.
    weights_source_file = h5py.File(weights_source_path, 'r')
    weights_destination_file = h5py.File(weights_destination_path)

    classifier_names = ['conv4_3_norm_mbox_conf',
                        'fc7_mbox_conf',
                        'conv6_2_mbox_conf',
                        'conv7_2_mbox_conf',
                        'conv8_2_mbox_conf',
                        'conv9_2_mbox_conf']

    conv4_3_norm_mbox_conf_kernel = weights_source_file[classifier_names[0]][classifier_names[0]]['kernel:0']
    conv4_3_norm_mbox_conf_bias = weights_source_file[classifier_names[0]][classifier_names[0]]['bias:0']

    print("Shape of the '{}' weights:".format(classifier_names[0]))
    print()
    print("kernel:\t", conv4_3_norm_mbox_conf_kernel.shape)
    print("bias:\t", conv4_3_norm_mbox_conf_bias.shape)

    n_classes_source = 81
    classes_of_interest = [0, 3, 8, 1, 2, 10, 4, 6, 12]

    subsampling_indices = []
    for i in range(int(324 / n_classes_source)):
        indices = np.array(classes_of_interest) + i * n_classes_source
        subsampling_indices.append(indices)
    subsampling_indices = list(np.concatenate(subsampling_indices))

    print(subsampling_indices)

    # TODO: Set the number of classes in the source weights file. Note that this number must include
    #       the background class, so for MS COCO's 80 classes, this must be 80 + 1 = 81.
    n_classes_source = 81
    # TODO: Set the indices of the classes that you want to pick for the sub-sampled weight tensors.
    #       In case you would like to just randomly sample a certain number of classes, you can just set
    #       `classes_of_interest` to an integer instead of the list below. Either way, don't forget to
    #       include the background class. That is, if you set an integer, and you want `n` positive classes,
    #       then you must set `classes_of_interest = n + 1`.
    classes_of_interest = [0, 3, 8, 1, 2, 10, 4, 6, 12]
    # classes_of_interest = 9 # Uncomment this in case you want to just randomly sub-sample the last axis instead of providing a list of indices.

    for name in classifier_names:
        # Get the trained weights for this layer from the source HDF5 weights file.
        kernel = weights_source_file[name][name]['kernel:0'].value
        bias = weights_source_file[name][name]['bias:0'].value

        # Get the shape of the kernel. We're interested in sub-sampling
        # the last dimension, 'o'.
        height, width, in_channels, out_channels = kernel.shape

        # Compute the indices of the elements we want to sub-sample.
        # Keep in mind that each classification predictor layer predicts multiple
        # bounding boxes for every spatial location, so we want to sub-sample
        # the relevant classes for each of these boxes.
        if isinstance(classes_of_interest, (list, tuple)):
            subsampling_indices = []
            for i in range(int(out_channels / n_classes_source)):
                indices = np.array(classes_of_interest) + i * n_classes_source
                subsampling_indices.append(indices)
            subsampling_indices = list(np.concatenate(subsampling_indices))
        elif isinstance(classes_of_interest, int):
            subsampling_indices = int(classes_of_interest * (out_channels / n_classes_source))
        else:
            raise ValueError("`classes_of_interest` must be either an integer or a list/tuple.")

        # Sub-sample the kernel and bias.
        # The `sample_tensors()` function used below provides extensive
        # documentation, so don't hesitate to read it if you want to know
        # what exactly is going on here.
        new_kernel, new_bias = sample_tensors(weights_list=[kernel, bias],
                                              sampling_instructions=[height, width, in_channels, subsampling_indices],
                                              axes=[[3]],
                                              # The one bias dimension corresponds to the last kernel dimension.
                                              init=['gaussian', 'zeros'],
                                              mean=0.0,
                                              stddev=0.005)

        # Delete the old weights from the destination file.
        del weights_destination_file[name][name]['kernel:0']
        del weights_destination_file[name][name]['bias:0']
        # Create new datasets for the sub-sampled weights.
        weights_destination_file[name][name].create_dataset(name='kernel:0', data=new_kernel)
        weights_destination_file[name][name].create_dataset(name='bias:0', data=new_bias)

    # Make sure all data is written to our output file before this sub-routine exits.
    weights_destination_file.flush()

    conv4_3_norm_mbox_conf_kernel = weights_destination_file[classifier_names[0]][classifier_names[0]]['kernel:0']
    conv4_3_norm_mbox_conf_bias = weights_destination_file[classifier_names[0]][classifier_names[0]]['bias:0']

    print("Shape of the '{}' weights:".format(classifier_names[0]))
    print()
    print("kernel:\t", conv4_3_norm_mbox_conf_kernel.shape)
    print("bias:\t", conv4_3_norm_mbox_conf_bias.shape)

    img_height = 300  # Height of the input images
    img_width = 300  # Width of the input images
    img_channels = 3  # Number of color channels of the input images
    subtract_mean = [123, 117, 104]  # The per-channel mean of the images in the dataset
    swap_channels = [2, 1,
                     0]  # The color channel order in the original SSD is BGR, so we should set this to `True`, but weirdly the results are better without swapping.
    # TODO: Set the number of classes.
    n_classes = 8  # Number of positive classes, e.g. 20 for Pascal VOC, 80 for MS COCO
    scales = [0.07, 0.15, 0.33, 0.51, 0.69, 0.87,
              1.05]  # The anchor box scaling factors used in the original SSD300 for the MS COCO datasets.
    # scales = [0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05] # The anchor box scaling factors used in the original SSD300 for the Pascal VOC datasets.
    aspect_ratios = [[1.0, 2.0, 0.5],
                     [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                     [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                     [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                     [1.0, 2.0, 0.5],
                     [1.0, 2.0, 0.5]]  # The anchor box aspect ratios used in the original SSD300; the order matters
    two_boxes_for_ar1 = True
    steps = [8, 16, 32, 64, 100,
             300]  # The space between two adjacent anchor box center points for each predictor layer.
    offsets = [0.5, 0.5, 0.5, 0.5, 0.5,
               0.5]  # The offsets of the first anchor box center points from the top and left borders of the image as a fraction of the step size for each predictor layer.
    clip_boxes = False  # Whether or not you want to limit the anchor boxes to lie entirely within the image boundaries
    variances = [0.1, 0.1, 0.2,
                 0.2]  # The variances by which the encoded target coordinates are scaled as in the original implementation
    normalize_coords = True

    # 1: Build the Keras model

    # K.clear_session()  # Clear previous models from memory.
    #
    # model = ssd_300(image_size=(img_height, img_width, img_channels),
    #                 n_classes=n_classes,
    #                 mode='inference',
    #                 l2_regularization=0.0005,
    #                 scales=scales,
    #                 aspect_ratios_per_layer=aspect_ratios,
    #                 two_boxes_for_ar1=two_boxes_for_ar1,
    #                 steps=steps,
    #                 offsets=offsets,
    #                 clip_boxes=clip_boxes,
    #                 variances=variances,
    #                 normalize_coords=normalize_coords,
    #                 subtract_mean=subtract_mean,
    #                 divide_by_stddev=None,
    #                 swap_channels=swap_channels,
    #                 confidence_thresh=0.5,
    #                 iou_threshold=0.45,
    #                 top_k=200,
    #                 nms_max_output_size=400,
    #                 return_predictor_sizes=False)
    #
    # print("Model built.")
    #
    # # 2: Load the sub-sampled weights into the model.
    #
    # # Load the weights that we've just created via sub-sampling.
    # weights_path = weights_destination_path
    #
    # model.load_weights(weights_path, by_name=True)
    #
    # print("Weights file loaded:", weights_path)
    #
    # # 3: Instantiate an Adam optimizer and the SSD loss function and compile the model.
    #
    # adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    #
    # ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)
    #
    # model.compile(optimizer=adam, loss=ssd_loss.compute_loss)
    #
    # model.save("{}/{}.h5".format("./models/", "VGG_coco_SSD_300x300_iter_400000_subsampled_8_classes"))

    # load model
    model_path = './models/VGG_coco_SSD_300x300_iter_400000_subsampled_8_classes.h5'

    # We need to create an SSDLoss object in order to pass that to the model loader.
    ssd_loss = SSDLoss(neg_pos_ratio=3, n_neg_min=0, alpha=1.0)

    K.clear_session()  # Clear previous models from memory.

    model = load_model(model_path, custom_objects={'AnchorBoxes': AnchorBoxes,
                                                   'L2Normalization': L2Normalization,
                                                   'DecodeDetections': DecodeDetections,
                                                   'compute_loss': ssd_loss.compute_loss})

    assert model is not None
    print("model loaded..")

    for n in range(56, 61):

        # load images
        orig_images = []  # Store the images here.
        input_images = []  # Store resized versions of the images here.

        # We'll only load one image in this example.
        img_path = './examples/PETS/frame_00{}.jpg'.format(n)

        orig_images.append(imread(img_path))
        img = image.load_img(img_path, target_size=(img_height, img_width))
        img = image.img_to_array(img)
        input_images.append(img)
        input_images = np.array(input_images)
        print("shape of input array : {}".format(input_images.shape))

        # make predictions
        y_pred = model.predict(input_images)

        confidence_threshold = 0.5
        y_pred_thresh = [y_pred[k][y_pred[k, :, 1] > confidence_threshold] for k in range(y_pred.shape[0])]
        print("y_pred : {}".format(y_pred))

        np.set_printoptions(precision=2, suppress=True, linewidth=90)
        print("Predicted boxes:\n")
        print('   class   conf xmin   ymin   xmax   ymax')
        print(y_pred_thresh[0])

        # visualize the predictions
        for box in y_pred_thresh[0]:
            # Transform the predicted bounding boxes for the 300x300 image to the original image dimensions.
            if int(box[0]) == 3 or int(box[0]) == 0:
                xmin = int(box[2] * orig_images[0].shape[1] / img_width)
                ymin = int(box[3] * orig_images[0].shape[0] / img_height)
                xmax = int(box[4] * orig_images[0].shape[1] / img_width)
                ymax = int(box[5] * orig_images[0].shape[0] / img_height)
                cv2.rectangle(orig_images[0], (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.cvtColor(orig_images[0], cv2.COLOR_BGR2RGB)
        cv2.imwrite("./examples/PETS/frame_00{}_rect.jpg".format(n), orig_images[0])

    # Display the image and draw the predicted boxes onto it.

    # Set the colors for the bounding boxes
    # colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
    # classes = ['background',
    #            'aeroplane', 'bicycle', 'bird', 'boat',
    #            'bottle', 'bus', 'car', 'cat',
    #            'chair', 'cow', 'diningtable', 'dog',
    #            'horse', 'motorbike', 'person', 'pottedplant',
    #            'sheep', 'sofa', 'train', 'tvmonitor']
    #
    # plt.figure(figsize=(20, 12))
    # plt.imshow(orig_images[0])
