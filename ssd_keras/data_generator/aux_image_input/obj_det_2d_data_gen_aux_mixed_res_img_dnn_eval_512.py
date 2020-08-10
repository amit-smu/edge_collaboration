'''
A data generator for 2D object detection.

Copyright (C) 2018 Pierluigi Ferrari

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''

from __future__ import division
import numpy as np
import inspect
from collections import defaultdict
import warnings
import sklearn.utils
from copy import deepcopy
from PIL import Image
import cv2
import csv
import os
import sys
from tqdm import tqdm, trange
from xml.etree import ElementTree
import random
import pickle
from models.keras_ssd512 import ssd_512
from keras_loss_function.keras_ssd_loss import SSDLoss
from keras import backend as K
from keras.optimizers import Adam
from sklearn.preprocessing import PolynomialFeatures
import sys

try:
    import h5py
except ImportError:
    warnings.warn("'h5py' module is missing. The fast HDF5 dataset option will be unavailable.")
try:
    import json
except ImportError:
    warnings.warn("'json' module is missing. The JSON-parser will be unavailable.")
try:
    from bs4 import BeautifulSoup
except ImportError:
    warnings.warn("'BeautifulSoup' module is missing. The XML-parser will be unavailable.")
try:
    import pickle
except ImportError:
    warnings.warn(
        "'pickle' module is missing. You won't be able to save parsed file lists and annotations as pickled files.")

from ssd_encoder_decoder.ssd_input_encoder import SSDInputEncoder
from data_generator.object_detection_2d_image_boxes_validation_utils import BoxFilter


class DegenerateBatchError(Exception):
    '''
    An exception class to be raised if a generated batch ends up being degenerate,
    e.g. if a generated batch is empty.
    '''
    pass


class DatasetError(Exception):
    '''
    An exception class to be raised if a anything is wrong with the dataset,
    in particular if you try to generate batches when no dataset was loaded.
    '''
    pass


class DataGenerator:
    '''
    A generator to generate batches of samples and corresponding labels indefinitely.

    Can shuffle the dataset consistently after each complete pass.

    Currently provides three methods to parse annotation data: A general-purpose CSV parser,
    an XML parser for the Pascal VOC datasets, and a JSON parser for the MS COCO datasets.
    If the annotations of your dataset are in a format that is not supported by these parsers,
    you could just add another parser method and still use this generator.

    Can perform image transformations for data conversion and data augmentation,
    for details please refer to the documentation of the `generate()` method.
    '''

    def __init__(self,
                 load_images_into_memory=False,
                 hdf5_dataset_path=None,
                 filenames=None,
                 filenames_type='text',
                 images_dir=None,
                 labels=None,
                 image_ids=None,
                 eval_neutral=None,
                 labels_output_format=('class_id', 'xmin', 'ymin', 'xmax', 'ymax'),
                 verbose=True,
                 N=1,
                 resolution=512,
                 ref_cam=1,
                 collab_cam=4,
                 test_dataset="PETS"):
        '''
        Initializes the data generator. You can either load a dataset directly here in the constructor,
        e.g. an HDF5 dataset, or you can use one of the parser methods to read in a dataset.

        Arguments:
            load_images_into_memory (bool, optional): If `True`, the entire dataset will be loaded into memory.
                This enables noticeably faster data generation than loading batches of images into memory ad hoc.
                Be sure that you have enough memory before you activate this option.
            hdf5_dataset_path (str, optional): The full file path of an HDF5 file that contains a dataset in the
                format that the `create_hdf5_dataset()` method produces. If you load such an HDF5 dataset, you
                don't need to use any of the parser methods anymore, the HDF5 dataset already contains all relevant
                data.
            filenames (string or list, optional): `None` or either a Python list/tuple or a string representing
                a filepath. If a list/tuple is passed, it must contain the file names (full paths) of the
                images to be used. Note that the list/tuple must contain the paths to the images,
                not the images themselves. If a filepath string is passed, it must point either to
                (1) a pickled file containing a list/tuple as described above. In this case the `filenames_type`
                argument must be set to `pickle`.
                Or
                (2) a text file. Each line of the text file contains the file name (basename of the file only,
                not the full directory path) to one image and nothing else. In this case the `filenames_type`
                argument must be set to `text` and you must pass the path to the directory that contains the
                images in `images_dir`.
            filenames_type (string, optional): In case a string is passed for `filenames`, this indicates what
                type of file `filenames` is. It can be either 'pickle' for a pickled file or 'text' for a
                plain text file.
            images_dir (string, optional): In case a text file is passed for `filenames`, the full paths to
                the images will be composed from `images_dir` and the names in the text file, i.e. this
                should be the directory that contains the images to which the text file refers.
                If `filenames_type` is not 'text', then this argument is irrelevant.
            labels (string or list, optional): `None` or either a Python list/tuple or a string representing
                the path to a pickled file containing a list/tuple. The list/tuple must contain Numpy arrays
                that represent the labels of the dataset.
            image_ids (string or list, optional): `None` or either a Python list/tuple or a string representing
                the path to a pickled file containing a list/tuple. The list/tuple must contain the image
                IDs of the images in the dataset.
            eval_neutral (string or list, optional): `None` or either a Python list/tuple or a string representing
                the path to a pickled file containing a list/tuple. The list/tuple must contain for each image
                a list that indicates for each ground truth object in the image whether that object is supposed
                to be treated as neutral during an evaluation.
            labels_output_format (list, optional): A list of five strings representing the desired order of the five
                items class ID, xmin, ymin, xmax, ymax in the generated ground truth data (if any). The expected
                strings are 'xmin', 'ymin', 'xmax', 'ymax', 'class_id'.
            verbose (bool, optional): If `True`, prints out the progress for some constructor operations that may
                take a bit longer.
        '''

        #############################################################################################

        self.N = N  # number of cameras collaborating for PETS dataset
        print("\nnumber of collaborating cameras :{}\n".format(self.N))
        self.img_type_prefix = "JPEGImages"

        self.resolution = resolution
        print("Resolution: {}\n".format(self.resolution))

        self.ref_cam = ref_cam
        self.collab_cam = collab_cam
        self.test_dataset = test_dataset
        print("test_dataset: {}".format(self.test_dataset))

        print("Loading single image object detector model..")
        # load DNN model
        self.single_img_model = self.initialize_model()
        assert self.single_img_model is not None
        print("model loaded..")

        # load regression model
        print("loading regression model..")
        if self.test_dataset == "WILDTRACK":
            self.load_regression_model_WT()
        elif self.test_dataset == "PETS":
            self.load_regression_model_PETS()
        else:
            print("wrong dataset specified..")
        print("regression model loaded..")

        #############################################################################################

        self.labels_output_format = labels_output_format
        self.labels_format = {'class_id': labels_output_format.index('class_id'),
                              'xmin': labels_output_format.index('xmin'),
                              'ymin': labels_output_format.index('ymin'),
                              'xmax': labels_output_format.index('xmax'),
                              'ymax': labels_output_format.index('ymax')}  # This dictionary is for internal use.

        self.dataset_size = 0  # As long as we haven't loaded anything yet, the dataset size is zero.
        self.load_images_into_memory = load_images_into_memory
        self.images = None  # The only way that this list will not stay `None` is if `load_images_into_memory == True`.

        # `self.filenames` is a list containing all file names of the image samples (full paths).
        # Note that it does not contain the actual image files themselves. This list is one of the outputs of the parser methods.
        # In case you are loading an HDF5 dataset, this list will be `None`.
        if not filenames is None:
            if isinstance(filenames, (list, tuple)):
                self.filenames = filenames
            elif isinstance(filenames, str):
                with open(filenames, 'rb') as f:
                    if filenames_type == 'pickle':
                        self.filenames = pickle.load(f)
                    elif filenames_type == 'text':
                        self.filenames = [os.path.join(images_dir, line.strip()) for line in f]
                    else:
                        raise ValueError("`filenames_type` can be either 'text' or 'pickle'.")
            else:
                raise ValueError(
                    "`filenames` must be either a Python list/tuple or a string representing a filepath (to a pickled or text file). The value you passed is neither of the two.")
            self.dataset_size = len(self.filenames)
            self.dataset_indices = np.arange(self.dataset_size, dtype=np.int32)
            if load_images_into_memory:
                self.images = []
                if verbose:
                    it = tqdm(self.filenames, desc='Loading images into memory', file=sys.stdout)
                else:
                    it = self.filenames
                for filename in it:
                    with Image.open(filename) as image:
                        self.images.append(np.array(image, dtype=np.uint8))
        else:
            self.filenames = None

        # In case ground truth is available, `self.labels` is a list containing for each image a list (or NumPy array)
        # of ground truth bounding boxes for that image.
        if not labels is None:
            if isinstance(labels, str):
                with open(labels, 'rb') as f:
                    self.labels = pickle.load(f)
            elif isinstance(labels, (list, tuple)):
                self.labels = labels
            else:
                raise ValueError(
                    "`labels` must be either a Python list/tuple or a string representing the path to a pickled file containing a list/tuple. The value you passed is neither of the two.")
        else:
            self.labels = None

        if not image_ids is None:
            if isinstance(image_ids, str):
                with open(image_ids, 'rb') as f:
                    self.image_ids = pickle.load(f)
            elif isinstance(image_ids, (list, tuple)):
                self.image_ids = image_ids
            else:
                raise ValueError(
                    "`image_ids` must be either a Python list/tuple or a string representing the path to a pickled file containing a list/tuple. The value you passed is neither of the two.")
        else:
            self.image_ids = None

        if not eval_neutral is None:
            if isinstance(eval_neutral, str):
                with open(eval_neutral, 'rb') as f:
                    self.eval_neutral = pickle.load(f)
            elif isinstance(eval_neutral, (list, tuple)):
                self.eval_neutral = eval_neutral
            else:
                raise ValueError(
                    "`image_ids` must be either a Python list/tuple or a string representing the path to a pickled file containing a list/tuple. The value you passed is neither of the two.")
        else:
            self.eval_neutral = None

        if not hdf5_dataset_path is None:
            self.hdf5_dataset_path = hdf5_dataset_path
            self.load_hdf5_dataset(verbose=verbose)
        else:
            self.hdf5_dataset = None

    def initialize_model(self):
        img_height = 512
        img_width = 512

        # ####################################     Load Weights/MODEL here    ##########################################
        # # 1: Build the Keras model
        # K.clear_session()  # Clear previous models from memory.
        single_img_model = ssd_512(image_size=(img_height, img_width, 3),
                                   n_classes=20,
                                   mode='inference',
                                   l2_regularization=0.0005,
                                   scales=[0.07, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05],
                                   # The scales for MS COCO are [0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05]
                                   aspect_ratios_per_layer=[[1.0, 2.0, 0.5],
                                                            [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                                            [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                                            [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                                            [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                                            [1.0, 2.0, 0.5],
                                                            [1.0, 2.0, 0.5]],
                                   two_boxes_for_ar1=True,
                                   steps=[8, 16, 32, 64, 128, 256, 512],
                                   offsets=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                                   clip_boxes=False,
                                   variances=[0.1, 0.1, 0.2, 0.2],
                                   normalize_coords=True,
                                   subtract_mean=[123, 117, 104],
                                   swap_channels=[2, 1, 0],
                                   confidence_thresh=0.5,
                                   iou_threshold=0.45,
                                   top_k=200,
                                   nms_max_output_size=400)

        # 2: Load the trained weights into the model.
        # TODO: Set the path of the trained weights.
        # weights_path = './single_img_models/ssd512_PETS+WT_person_180_epoch-179_loss-2.8713_val_loss-2.7450.h5'
        weights_path = './single_img_models/ssd512_PETS+WT_max_epoch_250_epoch-232_loss-2.9462_val_loss-3.2491.h5'
        print(weights_path + "\n")

        single_img_model.load_weights(weights_path, by_name=True)

        # 3: Compile the model so that Keras won't complain the next time you load it.

        adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

        ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)

        single_img_model.compile(optimizer=adam, loss=ssd_loss.compute_loss)

        return single_img_model

    def load_regression_model_WT(self):
        degree = 5
        src_cam = self.collab_cam
        dst_cam = self.ref_cam
        # model_file_path = "regression_models/poly_feature_linear_regression_deg_{}_interaction_false_cam_{}" \
        #     .format(degree, src_cam)
        model_file_path = "regression_models/WT/poly_feature_l_reg_deg_{}_inter_false_src_{}_dst_{}_full_img" \
            .format(degree, src_cam, dst_cam)
        print(model_file_path)
        print("degree: {}, src_cam: {}\n".format(degree, src_cam))
        self.reg_model = None
        with open(model_file_path, 'rb') as input_file:
            self.reg_model = pickle.load(input_file)
            assert self.reg_model is not None

    def load_regression_model_PETS(self):
        degree = 4
        src_cam = self.collab_cam
        dst_cam = self.ref_cam
        # model_file_path = "regression_models/poly_feature_linear_regression_deg_{}_interaction_false_cam_{}{}" \
        #     .format(degree, src_cam, self.ref_cam)
        model_file_path = "regression_models/PETS/poly_feature_l_reg_deg_{}_inter_false_src_{}_dst_{}_full_img" \
            .format(degree, src_cam, dst_cam)
        print(model_file_path)
        print("degree: {}, src_cam: {}, ref_cam :{}\n".format(degree, src_cam, self.ref_cam))
        self.reg_model = None
        with open(model_file_path, 'rb') as input_file:
            self.reg_model = pickle.load(input_file)
            assert self.reg_model is not None

    def load_hdf5_dataset(self, verbose=True):
        '''
        Loads an HDF5 dataset that is in the format that the `create_hdf5_dataset()` method
        produces.

        Arguments:
            verbose (bool, optional): If `True`, prints out the progress while loading
                the dataset.

        Returns:
            None.
        '''

        self.hdf5_dataset = h5py.File(self.hdf5_dataset_path, 'r')
        self.dataset_size = len(self.hdf5_dataset['images'])
        self.dataset_indices = np.arange(self.dataset_size,
                                         dtype=np.int32)  # Instead of shuffling the HDF5 dataset or images in memory, we will shuffle this index list.

        if self.load_images_into_memory:
            self.images = []
            if verbose:
                tr = trange(self.dataset_size, desc='Loading images into memory', file=sys.stdout)
            else:
                tr = range(self.dataset_size)
            for i in tr:
                self.images.append(self.hdf5_dataset['images'][i].reshape(self.hdf5_dataset['image_shapes'][i]))

        if self.hdf5_dataset.attrs['has_labels']:
            self.labels = []
            labels = self.hdf5_dataset['labels']
            label_shapes = self.hdf5_dataset['label_shapes']
            if verbose:
                tr = trange(self.dataset_size, desc='Loading labels', file=sys.stdout)
            else:
                tr = range(self.dataset_size)
            for i in tr:
                self.labels.append(labels[i].reshape(label_shapes[i]))

        if self.hdf5_dataset.attrs['has_image_ids']:
            self.image_ids = []
            image_ids = self.hdf5_dataset['image_ids']
            if verbose:
                tr = trange(self.dataset_size, desc='Loading image IDs', file=sys.stdout)
            else:
                tr = range(self.dataset_size)
            for i in tr:
                self.image_ids.append(image_ids[i])

        if self.hdf5_dataset.attrs['has_eval_neutral']:
            self.eval_neutral = []
            eval_neutral = self.hdf5_dataset['eval_neutral']
            if verbose:
                tr = trange(self.dataset_size, desc='Loading evaluation-neutrality annotations', file=sys.stdout)
            else:
                tr = range(self.dataset_size)
            for i in tr:
                self.eval_neutral.append(eval_neutral[i])

    def parse_csv(self,
                  images_dir,
                  labels_filename,
                  input_format,
                  include_classes='all',
                  random_sample=False,
                  ret=False,
                  verbose=True):
        '''
        Arguments:
            images_dir (str): The path to the directory that contains the images.
            labels_filename (str): The filepath to a CSV file that contains one ground truth bounding box per line
                and each line contains the following six items: image file name, class ID, xmin, xmax, ymin, ymax.
                The six items do not have to be in a specific order, but they must be the first six columns of
                each line. The order of these items in the CSV file must be specified in `input_format`.
                The class ID is an integer greater than zero. Class ID 0 is reserved for the background class.
                `xmin` and `xmax` are the left-most and right-most absolute horizontal coordinates of the box,
                `ymin` and `ymax` are the top-most and bottom-most absolute vertical coordinates of the box.
                The image name is expected to be just the name of the image file without the directory path
                at which the image is located.
            input_format (list): A list of six strings representing the order of the six items
                image file name, class ID, xmin, xmax, ymin, ymax in the input CSV file. The expected strings
                are 'image_name', 'xmin', 'xmax', 'ymin', 'ymax', 'class_id'.
            include_classes (list, optional): Either 'all' or a list of integers containing the class IDs that
                are to be included in the dataset. If 'all', all ground truth boxes will be included in the dataset.
            random_sample (float, optional): Either `False` or a float in `[0,1]`. If this is `False`, the
                full dataset will be used by the generator. If this is a float in `[0,1]`, a randomly sampled
                fraction of the dataset will be used, where `random_sample` is the fraction of the dataset
                to be used. For example, if `random_sample = 0.2`, 20 precent of the dataset will be randomly selected,
                the rest will be ommitted. The fraction refers to the number of images, not to the number
                of boxes, i.e. each image that will be added to the dataset will always be added with all
                of its boxes.
            ret (bool, optional): Whether or not to return the outputs of the parser.
            verbose (bool, optional): If `True`, prints out the progress for operations that may take a bit longer.

        Returns:
            None by default, optionally lists for whichever are available of images, image filenames, labels, and image IDs.
        '''

        # Set class members.
        self.images_dir = images_dir
        self.labels_filename = labels_filename
        self.input_format = input_format
        self.include_classes = include_classes

        # Before we begin, make sure that we have a labels_filename and an input_format
        if self.labels_filename is None or self.input_format is None:
            raise ValueError(
                "`labels_filename` and/or `input_format` have not been set yet. You need to pass them as arguments.")

        # Erase data that might have been parsed before
        self.filenames = []
        self.image_ids = []
        self.labels = []

        # First, just read in the CSV file lines and sort them.

        data = []

        with open(self.labels_filename, newline='') as csvfile:
            csvread = csv.reader(csvfile, delimiter=',')
            next(csvread)  # Skip the header row.
            for row in csvread:  # For every line (i.e for every bounding box) in the CSV file...
                if self.include_classes == 'all' or int(row[self.input_format.index(
                        'class_id')].strip()) in self.include_classes:  # If the class_id is among the classes that are to be included in the dataset...
                    box = []  # Store the box class and coordinates here
                    box.append(row[self.input_format.index(
                        'image_name')].strip())  # Select the image name column in the input format and append its content to `box`
                    for element in self.labels_output_format:  # For each element in the output format (where the elements are the class ID and the four box coordinates)...
                        box.append(int(row[self.input_format.index(
                            element)].strip()))  # ...select the respective column in the input format and append it to `box`.
                    data.append(box)

        data = sorted(data)  # The data needs to be sorted, otherwise the next step won't give the correct result

        # Now that we've made sure that the data is sorted by file names,
        # we can compile the actual samples and labels lists

        current_file = data[0][0]  # The current image for which we're collecting the ground truth boxes
        current_image_id = data[0][0].split('.')[
            0]  # The image ID will be the portion of the image name before the first dot.
        current_labels = []  # The list where we collect all ground truth boxes for a given image
        add_to_dataset = False
        for i, box in enumerate(data):

            if box[0] == current_file:  # If this box (i.e. this line of the CSV file) belongs to the current image file
                current_labels.append(box[1:])
                if i == len(data) - 1:  # If this is the last line of the CSV file
                    if random_sample:  # In case we're not using the full dataset, but a random sample of it.
                        p = np.random.uniform(0, 1)
                        if p >= (1 - random_sample):
                            self.labels.append(np.stack(current_labels, axis=0))
                            self.filenames.append(os.path.join(self.images_dir, current_file))
                            self.image_ids.append(current_image_id)
                    else:
                        self.labels.append(np.stack(current_labels, axis=0))
                        self.filenames.append(os.path.join(self.images_dir, current_file))
                        self.image_ids.append(current_image_id)
            else:  # If this box belongs to a new image file
                if random_sample:  # In case we're not using the full dataset, but a random sample of it.
                    p = np.random.uniform(0, 1)
                    if p >= (1 - random_sample):
                        self.labels.append(np.stack(current_labels, axis=0))
                        self.filenames.append(os.path.join(self.images_dir, current_file))
                        self.image_ids.append(current_image_id)
                else:
                    self.labels.append(np.stack(current_labels, axis=0))
                    self.filenames.append(os.path.join(self.images_dir, current_file))
                    self.image_ids.append(current_image_id)
                current_labels = []  # Reset the labels list because this is a new file.
                current_file = box[0]
                current_image_id = box[0].split('.')[0]
                current_labels.append(box[1:])
                if i == len(data) - 1:  # If this is the last line of the CSV file
                    if random_sample:  # In case we're not using the full dataset, but a random sample of it.
                        p = np.random.uniform(0, 1)
                        if p >= (1 - random_sample):
                            self.labels.append(np.stack(current_labels, axis=0))
                            self.filenames.append(os.path.join(self.images_dir, current_file))
                            self.image_ids.append(current_image_id)
                    else:
                        self.labels.append(np.stack(current_labels, axis=0))
                        self.filenames.append(os.path.join(self.images_dir, current_file))
                        self.image_ids.append(current_image_id)

        self.dataset_size = len(self.filenames)
        self.dataset_indices = np.arange(self.dataset_size, dtype=np.int32)
        if self.load_images_into_memory:
            self.images = []
            if verbose:
                it = tqdm(self.filenames, desc='Loading images into memory', file=sys.stdout)
            else:
                it = self.filenames
            for filename in it:
                with Image.open(filename) as image:
                    self.images.append(np.array(image, dtype=np.uint8))

        if ret:  # In case we want to return these
            return self.images, self.filenames, self.labels, self.image_ids

    def parse_xml(self,
                  images_dirs,
                  image_set_filenames,
                  annotations_dirs=[],
                  classes=['background',
                           'aeroplane', 'bicycle', 'bird', 'boat',
                           'bottle', 'bus', 'car', 'cat',
                           'chair', 'cow', 'diningtable', 'dog',
                           'horse', 'motorbike', 'person', 'pottedplant',
                           'sheep', 'sofa', 'train', 'tvmonitor'],
                  include_classes='all',
                  exclude_truncated=False,
                  exclude_difficult=False,
                  ret=False,
                  verbose=True):
        '''
        This is an XML parser for the Pascal VOC datasets. It might be applicable to other datasets with minor changes to
        the code, but in its current form it expects the data format and XML tags of the Pascal VOC datasets.

        Arguments:
            images_dirs (list): A list of strings, where each string is the path of a directory that
                contains images that are to be part of the dataset. This allows you to aggregate multiple datasets
                into one (e.g. one directory that contains the images for Pascal VOC 2007, another that contains
                the images for Pascal VOC 2012, etc.).
            image_set_filenames (list): A list of strings, where each string is the path of the text file with the image
                set to be loaded. Must be one file per image directory given. These text files define what images in the
                respective image directories are to be part of the dataset and simply contains one image ID per line
                and nothing else.
            annotations_dirs (list, optional): A list of strings, where each string is the path of a directory that
                contains the annotations (XML files) that belong to the images in the respective image directories given.
                The directories must contain one XML file per image and the name of an XML file must be the image ID
                of the image it belongs to. The content of the XML files must be in the Pascal VOC format.
            classes (list, optional): A list containing the names of the object classes as found in the
                `name` XML tags. Must include the class `background` as the first list item. The order of this list
                defines the class IDs.
            include_classes (list, optional): Either 'all' or a list of integers containing the class IDs that
                are to be included in the dataset. If 'all', all ground truth boxes will be included in the dataset.
            exclude_truncated (bool, optional): If `True`, excludes boxes that are labeled as 'truncated'.
            exclude_difficult (bool, optional): If `True`, excludes boxes that are labeled as 'difficult'.
            ret (bool, optional): Whether or not to return the outputs of the parser.
            verbose (bool, optional): If `True`, prints out the progress for operations that may take a bit longer.

        Returns:
            None by default, optionally lists for whichever are available of images, image filenames, labels, image IDs,
            and a list indicating which boxes are annotated with the label "difficult".
        '''
        # Set class members.
        self.images_dirs = images_dirs
        self.annotations_dirs = annotations_dirs
        self.image_set_filenames = image_set_filenames
        self.classes = classes
        self.include_classes = include_classes

        # Erase data that might have been parsed before.
        self.filenames = []
        self.image_ids = []
        self.labels = []
        self.eval_neutral = []
        if not annotations_dirs:
            self.labels = None
            self.eval_neutral = None
            annotations_dirs = [None] * len(images_dirs)

        for images_dir, image_set_filename, annotations_dir in zip(images_dirs, image_set_filenames, annotations_dirs):
            # Read the image set file that so that we know all the IDs of all the images to be included in the dataset.
            with open(image_set_filename) as f:
                image_ids = [line.strip() for line in f]  # Note: These are strings, not integers.
                self.image_ids += image_ids

            if verbose:
                it = tqdm(image_ids, desc="Processing image set '{}'".format(os.path.basename(image_set_filename)),
                          file=sys.stdout)
            else:
                it = image_ids

            # ########################################################################################################
            # PETS or WILDTRAck
            if images_dir.__contains__("PNG"):
                # WILDTRACK
                img_type = "png"
                self.img_type_prefix = "PNGImages"
            else:
                # PETS
                img_type = "jpg"
                self.img_type_prefix = "JPEGImages"
            # ########################################################################################################

            # Loop over all images in this dataset.
            for image_id in it:

                filename = '{}'.format(image_id) + '.{}'.format(img_type)
                # filename = '{}'.format(image_id) + '.png'
                self.filenames.append(os.path.join(images_dir, filename))

                if not annotations_dir is None:
                    # Parse the XML file for this image.
                    with open(os.path.join(annotations_dir, image_id + '.xml')) as f:
                        soup = BeautifulSoup(f, 'xml')

                    folder = soup.folder.text  # In case we want to return the folder in addition to the image file name. Relevant for determining which dataset an image belongs to.
                    # filename = soup.filename.text

                    boxes = []  # We'll store all boxes for this image here.
                    eval_neutr = []  # We'll store whether a box is annotated as "difficult" here.
                    objects = soup.find_all('object')  # Get a list of all objects in this image.

                    # Parse the data for each object.
                    for obj in objects:
                        class_name = obj.find('name', recursive=False).text
                        class_id = self.classes.index(class_name)
                        # Check whether this class is supposed to be included in the dataset.
                        if (not self.include_classes == 'all') and (not class_id in self.include_classes): continue
                        pose = obj.find('pose', recursive=False).text
                        truncated = int(obj.find('truncated', recursive=False).text)
                        if exclude_truncated and (truncated == 1): continue
                        difficult = int(obj.find('difficult', recursive=False).text)
                        if exclude_difficult and (difficult == 1): continue
                        # Get the bounding box coordinates.
                        bndbox = obj.find('bndbox', recursive=False)
                        xmin = int(bndbox.xmin.text)
                        ymin = int(bndbox.ymin.text)
                        xmax = int(bndbox.xmax.text)
                        ymax = int(bndbox.ymax.text)
                        item_dict = {'folder': folder,
                                     'image_name': filename,
                                     'image_id': image_id,
                                     'class_name': class_name,
                                     'class_id': class_id,
                                     'pose': pose,
                                     'truncated': truncated,
                                     'difficult': difficult,
                                     'xmin': xmin,
                                     'ymin': ymin,
                                     'xmax': xmax,
                                     'ymax': ymax}
                        box = []
                        for item in self.labels_output_format:
                            box.append(item_dict[item])
                        boxes.append(box)
                        if difficult:
                            eval_neutr.append(True)
                        else:
                            eval_neutr.append(False)

                    self.labels.append(boxes)
                    self.eval_neutral.append(eval_neutr)

        self.dataset_size = len(self.filenames)
        self.dataset_indices = np.arange(self.dataset_size, dtype=np.int32)
        if self.load_images_into_memory:
            self.images = []
            if verbose:
                it = tqdm(self.filenames, desc='Loading images into memory', file=sys.stdout)
            else:
                it = self.filenames
            for filename in it:
                with Image.open(filename) as image:
                    self.images.append(np.array(image, dtype=np.uint8))

        if ret:
            return self.images, self.filenames, self.labels, self.image_ids, self.eval_neutral

    def parse_json(self,
                   images_dirs,
                   annotations_filenames,
                   ground_truth_available=False,
                   include_classes='all',
                   ret=False,
                   verbose=True):
        '''
        This is an JSON parser for the MS COCO datasets. It might be applicable to other datasets with minor changes to
        the code, but in its current form it expects the JSON format of the MS COCO datasets.

        Arguments:
            images_dirs (list, optional): A list of strings, where each string is the path of a directory that
                contains images that are to be part of the dataset. This allows you to aggregate multiple datasets
                into one (e.g. one directory that contains the images for MS COCO Train 2014, another one for MS COCO
                Val 2014, another one for MS COCO Train 2017 etc.).
            annotations_filenames (list): A list of strings, where each string is the path of the JSON file
                that contains the annotations for the images in the respective image directories given, i.e. one
                JSON file per image directory that contains the annotations for all images in that directory.
                The content of the JSON files must be in MS COCO object detection format. Note that these annotations
                files do not necessarily need to contain ground truth information. MS COCO also provides annotations
                files without ground truth information for the test datasets, called `image_info_[...].json`.
            ground_truth_available (bool, optional): Set `True` if the annotations files contain ground truth information.
            include_classes (list, optional): Either 'all' or a list of integers containing the class IDs that
                are to be included in the dataset. If 'all', all ground truth boxes will be included in the dataset.
            ret (bool, optional): Whether or not to return the outputs of the parser.
            verbose (bool, optional): If `True`, prints out the progress for operations that may take a bit longer.

        Returns:
            None by default, optionally lists for whichever are available of images, image filenames, labels and image IDs.
        '''
        self.images_dirs = images_dirs
        self.annotations_filenames = annotations_filenames
        self.include_classes = include_classes
        # Erase data that might have been parsed before.
        self.filenames = []
        self.image_ids = []
        self.labels = []
        if not ground_truth_available:
            self.labels = None

        # Build the dictionaries that map between class names and class IDs.
        with open(annotations_filenames[0], 'r') as f:
            annotations = json.load(f)
        # Unfortunately the 80 MS COCO class IDs are not all consecutive. They go
        # from 1 to 90 and some numbers are skipped. Since the IDs that we feed
        # into a neural network must be consecutive, we'll save both the original
        # (non-consecutive) IDs as well as transformed maps.
        # We'll save both the map between the original
        self.cats_to_names = {}  # The map between class names (values) and their original IDs (keys)
        self.classes_to_names = []  # A list of the class names with their indices representing the transformed IDs
        self.classes_to_names.append(
            'background')  # Need to add the background class first so that the indexing is right.
        self.cats_to_classes = {}  # A dictionary that maps between the original (keys) and the transformed IDs (values)
        self.classes_to_cats = {}  # A dictionary that maps between the transformed (keys) and the original IDs (values)
        for i, cat in enumerate(annotations['categories']):
            self.cats_to_names[cat['id']] = cat['name']
            self.classes_to_names.append(cat['name'])
            self.cats_to_classes[cat['id']] = i + 1
            self.classes_to_cats[i + 1] = cat['id']

        # Iterate over all datasets.
        for images_dir, annotations_filename in zip(self.images_dirs, self.annotations_filenames):
            # Load the JSON file.
            with open(annotations_filename, 'r') as f:
                annotations = json.load(f)

            if ground_truth_available:
                # Create the annotations map, a dictionary whose keys are the image IDs
                # and whose values are the annotations for the respective image ID.
                image_ids_to_annotations = defaultdict(list)
                for annotation in annotations['annotations']:
                    image_ids_to_annotations[annotation['image_id']].append(annotation)

            if verbose:
                it = tqdm(annotations['images'], desc="Processing '{}'".format(os.path.basename(annotations_filename)),
                          file=sys.stdout)
            else:
                it = annotations['images']

            # Loop over all images in this dataset.
            for img in it:

                self.filenames.append(os.path.join(images_dir, img['file_name']))
                self.image_ids.append(img['id'])

                if ground_truth_available:
                    # Get all annotations for this image.
                    annotations = image_ids_to_annotations[img['id']]
                    boxes = []
                    for annotation in annotations:
                        cat_id = annotation['category_id']
                        # Check if this class is supposed to be included in the dataset.
                        if (not self.include_classes == 'all') and (not cat_id in self.include_classes): continue
                        # Transform the original class ID to fit in the sequence of consecutive IDs.
                        class_id = self.cats_to_classes[cat_id]
                        xmin = annotation['bbox'][0]
                        ymin = annotation['bbox'][1]
                        width = annotation['bbox'][2]
                        height = annotation['bbox'][3]
                        # Compute `xmax` and `ymax`.
                        xmax = xmin + width
                        ymax = ymin + height
                        item_dict = {'image_name': img['file_name'],
                                     'image_id': img['id'],
                                     'class_id': class_id,
                                     'xmin': xmin,
                                     'ymin': ymin,
                                     'xmax': xmax,
                                     'ymax': ymax}
                        box = []
                        for item in self.labels_output_format:
                            box.append(item_dict[item])
                        boxes.append(box)
                    self.labels.append(boxes)

        self.dataset_size = len(self.filenames)
        self.dataset_indices = np.arange(self.dataset_size, dtype=np.int32)
        if self.load_images_into_memory:
            self.images = []
            if verbose:
                it = tqdm(self.filenames, desc='Loading images into memory', file=sys.stdout)
            else:
                it = self.filenames
            for filename in it:
                with Image.open(filename) as image:
                    self.images.append(np.array(image, dtype=np.uint8))

        if ret:
            return self.images, self.filenames, self.labels, self.image_ids

    def create_hdf5_dataset(self,
                            file_path='dataset.h5',
                            resize=False,
                            variable_image_size=True,
                            verbose=True):
        '''
        Converts the currently loaded dataset into a HDF5 file. This HDF5 file contains all
        images as uncompressed arrays in a contiguous block of memory, which allows for them
        to be loaded faster. Such an uncompressed dataset, however, may take up considerably
        more space on your hard drive than the sum of the source images in a compressed format
        such as JPG or PNG.

        It is recommended that you always convert the dataset into an HDF5 dataset if you
        have enugh hard drive space since loading from an HDF5 dataset accelerates the data
        generation noticeably.

        Note that you must load a dataset (e.g. via one of the parser methods) before creating
        an HDF5 dataset from it.

        The created HDF5 dataset will remain open upon its creation so that it can be used right
        away.

        Arguments:
            file_path (str, optional): The full file path under which to store the HDF5 dataset.
                You can load this output file via the `DataGenerator` constructor in the future.
            resize (tuple, optional): `False` or a 2-tuple `(height, width)` that represents the
                target size for the images. All images in the dataset will be resized to this
                target size before they will be written to the HDF5 file. If `False`, no resizing
                will be performed.
            variable_image_size (bool, optional): The only purpose of this argument is that its
                value will be stored in the HDF5 dataset in order to be able to quickly find out
                whether the images in the dataset all have the same size or not.
            verbose (bool, optional): Whether or not prit out the progress of the dataset creation.

        Returns:
            None.
        '''

        self.hdf5_dataset_path = file_path

        dataset_size = len(self.filenames)

        # Create the HDF5 file.
        hdf5_dataset = h5py.File(file_path, 'w')

        # Create a few attributes that tell us what this dataset contains.
        # The dataset will obviously always contain images, but maybe it will
        # also contain labels, image IDs, etc.
        hdf5_dataset.attrs.create(name='has_labels', data=False, shape=None, dtype=np.bool_)
        hdf5_dataset.attrs.create(name='has_image_ids', data=False, shape=None, dtype=np.bool_)
        hdf5_dataset.attrs.create(name='has_eval_neutral', data=False, shape=None, dtype=np.bool_)
        # It's useful to be able to quickly check whether the images in a dataset all
        # have the same size or not, so add a boolean attribute for that.
        if variable_image_size and not resize:
            hdf5_dataset.attrs.create(name='variable_image_size', data=True, shape=None, dtype=np.bool_)
        else:
            hdf5_dataset.attrs.create(name='variable_image_size', data=False, shape=None, dtype=np.bool_)

        # Create the dataset in which the images will be stored as flattened arrays.
        # This allows us, among other things, to store images of variable size.
        hdf5_images = hdf5_dataset.create_dataset(name='images',
                                                  shape=(dataset_size,),
                                                  maxshape=(None),
                                                  dtype=h5py.special_dtype(vlen=np.uint8))

        # Create the dataset that will hold the image heights, widths and channels that
        # we need in order to reconstruct the images from the flattened arrays later.
        hdf5_image_shapes = hdf5_dataset.create_dataset(name='image_shapes',
                                                        shape=(dataset_size, 3),
                                                        maxshape=(None, 3),
                                                        dtype=np.int32)

        if not (self.labels is None):
            # Create the dataset in which the labels will be stored as flattened arrays.
            hdf5_labels = hdf5_dataset.create_dataset(name='labels',
                                                      shape=(dataset_size,),
                                                      maxshape=(None),
                                                      dtype=h5py.special_dtype(vlen=np.int32))

            # Create the dataset that will hold the dimensions of the labels arrays for
            # each image so that we can restore the labels from the flattened arrays later.
            hdf5_label_shapes = hdf5_dataset.create_dataset(name='label_shapes',
                                                            shape=(dataset_size, 2),
                                                            maxshape=(None, 2),
                                                            dtype=np.int32)

            hdf5_dataset.attrs.modify(name='has_labels', value=True)

        if not (self.image_ids is None):
            hdf5_image_ids = hdf5_dataset.create_dataset(name='image_ids',
                                                         shape=(dataset_size,),
                                                         maxshape=(None),
                                                         dtype=h5py.special_dtype(vlen=str))

            hdf5_dataset.attrs.modify(name='has_image_ids', value=True)

        if not (self.eval_neutral is None):
            # Create the dataset in which the labels will be stored as flattened arrays.
            hdf5_eval_neutral = hdf5_dataset.create_dataset(name='eval_neutral',
                                                            shape=(dataset_size,),
                                                            maxshape=(None),
                                                            dtype=h5py.special_dtype(vlen=np.bool_))

            hdf5_dataset.attrs.modify(name='has_eval_neutral', value=True)

        if verbose:
            tr = trange(dataset_size, desc='Creating HDF5 dataset', file=sys.stdout)
        else:
            tr = range(dataset_size)

        # Iterate over all images in the dataset.
        for i in tr:

            # Store the image.
            with Image.open(self.filenames[i]) as image:

                image = np.asarray(image, dtype=np.uint8)

                # Make sure all images end up having three channels.
                if image.ndim == 2:
                    image = np.stack([image] * 3, axis=-1)
                elif image.ndim == 3:
                    if image.shape[2] == 1:
                        image = np.concatenate([image] * 3, axis=-1)
                    elif image.shape[2] == 4:
                        image = image[:, :, :3]

                if resize:
                    image = cv2.resize(image, dsize=(resize[1], resize[0]))

                # Flatten the image array and write it to the images dataset.
                hdf5_images[i] = image.reshape(-1)
                # Write the image's shape to the image shapes dataset.
                hdf5_image_shapes[i] = image.shape

            # Store the ground truth if we have any.
            if not (self.labels is None):
                labels = np.asarray(self.labels[i])
                # Flatten the labels array and write it to the labels dataset.
                hdf5_labels[i] = labels.reshape(-1)
                # Write the labels' shape to the label shapes dataset.
                hdf5_label_shapes[i] = labels.shape

            # Store the image ID if we have one.
            if not (self.image_ids is None):
                hdf5_image_ids[i] = self.image_ids[i]

            # Store the evaluation-neutrality annotations if we have any.
            if not (self.eval_neutral is None):
                hdf5_eval_neutral[i] = self.eval_neutral[i]

        hdf5_dataset.close()
        self.hdf5_dataset = h5py.File(file_path, 'r')
        self.hdf5_dataset_path = file_path
        self.dataset_size = len(self.hdf5_dataset['images'])
        self.dataset_indices = np.arange(self.dataset_size,
                                         dtype=np.int32)  # Instead of shuffling the HDF5 dataset, we will shuffle this index list.

    def map_coordinates_WT(self, org_objects):
        # print("Len of org_objects: {}".format(len(org_objects)))
        img_res = (1920, 1080)
        degree = 5
        poly_features = PolynomialFeatures(degree=degree, interaction_only=False)
        # convert objct coordinates to full WT image (regression was trained on full image coordinates)
        # gt_overlap_4_1 = (1089, 6, 1914, 1071)
        # gt_overlap_1_4 = (6, 4, 908, 1074)  # projected on view 1
        #
        # for obj in org_objects:
        #     obj[2] = obj[2] + gt_overlap_4_1[0]
        #     obj[3] = obj[3] + gt_overlap_4_1[1]
        #     obj[4] = obj[4] + gt_overlap_4_1[0]
        #     obj[5] = obj[5] + gt_overlap_4_1[1]

        # ################ TEst ############################################
        # collab_img = cv2.imread("../dataset/Wildtrack_dataset/PNGImages/{}".format(collab_img_id))
        # # print(ref_img_id)
        # ref_img = cv2.imread("../dataset/Wildtrack_dataset/PNGImages/{}.png".format(ref_img_id))
        # assert collab_img is not None
        # assert ref_img is not None

        # for obj in org_objects:
        #     cv2.rectangle(collab_img, (obj[2], obj[3]), (obj[4], obj[5]), (0, 0, 255), 2)
        # cv2.imwrite("temp/{}".format(collab_img_id), collab_img)

        # remove class id and conf score from detected objcts
        X = []
        for obj in org_objects:
            xmin, ymin, xmax, ymax = int(obj[2]), int(obj[3]), int(obj[4]), int(obj[5])
            src_width = xmax - xmin
            src_height = ymax - ymin
            src_bt_mid_x = xmin + (src_width // 2)
            X.append([src_bt_mid_x, ymax, src_width, src_height])

        X = np.array(X)
        X_poly = poly_features.fit_transform(X)
        # map coordinates
        mapped_objects = []
        for index, row in enumerate(X_poly):
            row = np.reshape(row, newshape=(1, -1))
            row_pred = self.reg_model.predict(row)
            dst_bt_mid_x, dst_ymax, dst_width, dst_height = np.array(row_pred[0], dtype=np.int32)
            # compute coordinates from mid point and width height
            d_xmin = dst_bt_mid_x - (dst_width // 2)
            d_ymin = dst_ymax - dst_height
            d_xmax = dst_bt_mid_x + (dst_width // 2)
            # draw mapped coordinteas on referce image
            # cv2.rectangle(ref_img, (d_xmin, d_ymin), (d_xmax, dst_ymax), (0, 255, 0), 2)
            # convert mapped coordintse from full image coords to sub-sample image coords
            # d_xmin = d_xmin - gt_overlap_1_4[0]
            # d_ymin = d_ymin - gt_overlap_1_4[1]
            # d_xmax = d_xmax - gt_overlap_1_4[0]
            # dst_ymax = dst_ymax - gt_overlap_1_4[1]

            # rescale coordintes to 512x512 image
            # width = gt_overlap_1_4[2] - gt_overlap_1_4[0]
            # height = gt_overlap_1_4[3] - gt_overlap_1_4[1]
            # d_xmin = int((d_xmin / img_res[0]) * 512.0)
            # d_ymin = int((d_ymin / img_res[1]) * 512.0)
            # d_xmax = int((d_xmax / img_res[0]) * 512.0)
            # dst_ymax = int((dst_ymax / img_res[1]) * 512.0)
            mapped_objects.append([(org_objects[index])[0], (org_objects[index])[1], d_xmin, d_ymin, d_xmax, dst_ymax])

        # cv2.imwrite("temp/{}.jpg".format(ref_img_id), ref_img)
        return mapped_objects

    def map_coordinates_PETS(self, org_objects):
        # print("Len of org_objects: {}".format(len(org_objects)))
        degree = 4
        poly_features = PolynomialFeatures(degree=degree, interaction_only=False)

        # convert objct coordinates to full PETS image (regression was trained on full image coordinates)
        # gt_overlap_c_r = (155, 92, 720, 516)
        # gt_overlap_c_r = (0, 0, 0, 0)
        # gt_overlap_r_c = (28, 101, 617, 492)  # projected on view 7

        # dimensions of overlap area in reference camra
        # olap_rf_area_w = float(gt_overlap_r_c[2] - gt_overlap_r_c[0])  # ref cam's overlap area width
        # olap_rf_area_h = float(gt_overlap_r_c[3] - gt_overlap_r_c[1])

        PETS_org_size = (720, 576)  # w,h
        # intermediate_width = float(PETS_org_size[0] - gt_overlap_r_c[0])
        # intermediate_height = float(PETS_org_size[1] - gt_overlap_r_c[1])

        # for obj in org_objects:
        #     obj[2] = obj[2] + gt_overlap_c_r[0]
        #     obj[3] = obj[3] + gt_overlap_c_r[1]
        #     obj[4] = obj[4] + gt_overlap_c_r[0]
        #     obj[5] = obj[5] + gt_overlap_c_r[1]

        # ############## Test ##########################
        # collab_img = cv2.imread("../dataset/PETS_org/JPEGImages_cropped_r7_c8_8/{}".format(collab_img_id))
        # print(ref_img_id)
        # ref_img = cv2.imread("../dataset/PETS_org/JPEGImages_cropped_r7_c8_7/{}.jpg".format(ref_img_id))
        # assert collab_img is not None
        # assert ref_img is not None

        # for obj in org_objects:
        #     cv2.rectangle(collab_img, (obj[2], obj[3]), (obj[4], obj[5]), (0, 0, 255), 2)
        # cv2.imwrite("temp/{}".format(collab_img_id), collab_img)
        # ###############################################

        # remove class id and conf score from detected objcts
        X = []
        for obj in org_objects:
            xmin, ymin, xmax, ymax = int(obj[2]), int(obj[3]), int(obj[4]), int(obj[5])
            src_width = xmax - xmin
            src_height = ymax - ymin
            src_bt_mid_x = xmin + (src_width // 2)
            X.append([src_bt_mid_x, ymax, src_width, src_height])

        X = np.array(X)
        X_poly = poly_features.fit_transform(X)

        # map coordinates
        mapped_objects = []
        for index, row in enumerate(X_poly):
            row = np.reshape(row, newshape=(1, -1))
            row_pred = self.reg_model.predict(row)
            dst_bt_mid_x, dst_ymax, dst_width, dst_height = np.array(row_pred[0], dtype=np.int32)
            # compute coordinates from mid point and width height
            d_xmin = dst_bt_mid_x - (dst_width // 2)
            d_ymin = dst_ymax - dst_height
            d_xmax = dst_bt_mid_x + (dst_width // 2)

            # draw mapped coordinates on reference image
            # cv2.rectangle(ref_img, (d_xmin, d_ymin), (d_xmax, dst_ymax), (0, 255, 0), 2)

            # convert mapped coordintse from full image coords to overlap area coords of reference camera
            # d_xmin = d_xmin - gt_overlap_r_c[0]
            # d_ymin = d_ymin - gt_overlap_r_c[1]
            # d_xmax = d_xmax - gt_overlap_r_c[0]
            # dst_ymax = dst_ymax - gt_overlap_r_c[1]
            #
            # # scale coordinates to size of org overlap area size in reference camera
            # d_xmin = int(d_xmin * (olap_rf_area_w / intermediate_width))
            # d_ymin = int(d_ymin * (olap_rf_area_h / intermediate_height))
            # d_xmax = int(d_xmax * (olap_rf_area_w / intermediate_width))
            # dst_ymax = int(dst_ymax * (olap_rf_area_h / intermediate_height))

            # scale coords to 512x512 (ref cam imge is scaled to this res by data generator)
            # d_xmin = int(d_xmin * (512.0 / olap_rf_area_w))
            # d_ymin = int(d_ymin * (512.0 / olap_rf_area_h))
            # d_xmax = int(d_xmax * (512.0 / olap_rf_area_w))
            # dst_ymax = int(dst_ymax * (512.0 / olap_rf_area_h))

            mapped_objects.append([(org_objects[index])[0], (org_objects[index])[1], d_xmin, d_ymin, d_xmax, dst_ymax])
        # write image to file
        # cv2.imwrite("temp/{}.jpg".format(ref_img_id), ref_img)

        return mapped_objects

    def get_aux_channels_batch_darknet_randomization(self, batch_X_data, batch_y_data, randomize):
        """
        create randomization as done during darknet training
        """
        img_height = 512
        img_width = 512
        batch_y_auxillary = []

        for index, img_annot in enumerate(batch_y_data):
            aux_channel = np.full((img_height, img_width, 1), 114, dtype=np.uint8)

            total_objs = len(img_annot)

            for i, obj in enumerate(img_annot):
                # ignore X% of boxes (no prior for them)
                r_int_1 = np.random.randint(1, 10)
                if r_int_1 > 8:
                    continue
                # add remaining objects to the prior
                class_id, xmin, ymin, xmax, ymax = obj
                if randomize:
                    width_rnum = np.random.uniform(-0.20, 0.20)
                    height_rnum = np.random.uniform(-0.20, 0.20)

                    obj_width = xmax - xmin
                    obj_height = ymax - ymin

                    xmin = int(xmin + (obj_width * width_rnum))
                    ymin = int(ymin + (obj_height * height_rnum))
                    xmax = int(xmin + obj_width + (obj_width * width_rnum))
                    ymax = int(ymin + obj_height + (obj_height * height_rnum))

                    # check for out of frame values
                    xmin = max(0, xmin)
                    ymin = max(0, ymin)
                    xmax = min(img_width, xmax)
                    ymax = min(img_height, ymax)

                    aux_channel[ymin:ymax, xmin:xmax] = 255

                else:
                    aux_channel[ymin:ymax, xmin:xmax] = 255
            batch_y_auxillary.append(aux_channel)

        return np.array(batch_y_auxillary)

    def get_gt_objects_WT(self, img_name, WT_annotation_dir):
        """
        get the ground truth objects using given image path
        :param img_path:
        :return:
        """
        objects = []

        annot_name = "{}.xml".format(img_name)
        annot_path = "{}/{}".format(WT_annotation_dir, annot_name)
        root = ElementTree.parse(annot_path).getroot()
        persons = root.findall('object')
        for p in persons:
            bndbox = p.find('bndbox')
            xmin = int(bndbox[0].text)
            ymin = int(bndbox[1].text)
            xmax = int(bndbox[2].text)
            ymax = int(bndbox[3].text)
            objects.append([15, 1, xmin, ymin, xmax, ymax])
            # print(xmin, ymin, xmax, ymax)
        return objects

    def get_aux_channel_detected_boxes(self, batch_file_names, batch_img_ids):
        """
        create mask using detected bounding boxes from full size images
        :return:
        """
        DUMP_DATA = False
        shared_reg_bbox = []  # xmin, ymin, xmax, ymax of shared region (drawn on collaborating cam)
        ICOV_TH = 0.10  # min icov value to select an object
        batch_x_prior = []
        for i in range(len(batch_file_names)):
            aux_channel = np.full((512, 512, 1), 114, dtype=np.uint8)
            if self.img_type_prefix == "PNGImages":  # WILDTRACK
                img_id = batch_img_ids[i]
                collab_img_id = "C{}_{}.png".format(self.collab_cam, img_id[3:])
                file_name = batch_file_names[i]
                # print(file_name)
                # print(img_id)
                img_name_len = len(img_id) + 4
                file_name = file_name[:-1 * img_name_len]
                # file_name = "{}/{}".format(file_name, self.collab_cam)
                collab_file_path = "{}/{}".format(file_name, collab_img_id)
                # shared_reg_bbox = [1089, 6, 1914, 1071]  # gt for camera 1, 4
                shared_reg_bbox = [197, 146, 1467, 1040]  # for camera 5, 7
                # shared_reg_bbox = [203, 202, 1719, 981]  # gt for camera 6, 1
                annot_dir = "../dataset/Wildtrack_dataset/Annotations"

            elif self.img_type_prefix == "JPEGImages":  # PETS dataset
                img_id = batch_img_ids[i]
                collab_img_id = "frame_{}_{}.jpg".format(self.collab_cam, img_id[8:])
                file_name = batch_file_names[i]
                img_name_len = len(img_id) + 4
                file_name = file_name[:-1 * img_name_len]
                # file_name = "{}_{}".format(file_name, self.collab_cam)
                collab_file_path = "{}/{}".format(file_name, collab_img_id)
                # print("{}, {}, {}\n".format(batch_file_names[i], batch_img_ids[i], collab_file_path))
                # shared_reg_bbox = [155, 92, 720, 516]  # in collab cam perspective (cam 7 , 8)
                shared_reg_bbox = [128, 104, 694, 520]  # in collab cam perspective (cam 8, 5) 
                # shared_reg_bbox = [21, 100, 571, 493]  # in collab cam perspective (cam 5, 7) 
                annot_dir = "../dataset/PETS_org/Annotations"
            # read image file
            # print(img_id, collab_file_path)
            # print(collab_file_path)
            collab_img = cv2.imread(collab_file_path)
            assert collab_img is not None
            # collab_img = cv2.resize(collab_img, dsize=(512, 512), interpolation=cv2.INTER_CUBIC)
            objects = self.detect_objects(collab_img)   
            # objects = self.detect_objects(cv2.imread(batch_file_names[i]))
            # objects = self.get_gt_objects_WT(collab_img_id[:-4], annot_dir)
            # objects = self.get_gt_objects_WT(batch_img_ids[i], annot_dir)
            if len(objects) == 0:
                batch_x_prior.append(aux_channel)
                continue
            # if DUMP_DATA:
            #     for obj in objects:
            #         cv2.rectangle(collab_img, (obj[2], obj[3]), (obj[4], obj[5]), (255, 0, 0), 2)
            #         cv2.imwrite("temp/{}".format(collab_img_id), collab_img)

            # map coordinates to reference camera coordinate system
            # select objects in shared region
            shared_reg_objects = []
            for obj in objects:
                obj_bbox = [obj[2], obj[3], obj[4], obj[5]]
                icov = self.bb_icov(obj_bbox, shared_reg_bbox)
                if icov >= ICOV_TH:
                    shared_reg_objects.append(obj)

            if len(shared_reg_objects) == 0:
                batch_x_prior.append(aux_channel)
                continue

            # map coordnates to other camera view
            if self.img_type_prefix == "PNGImages":  # WILDTRACK
                mapped_objects = self.map_coordinates_WT(shared_reg_objects)
            elif self.img_type_prefix == "JPEGImages":  # WILDTRACK
                # objects = self.map_coordinates_PETS(objects, collab_img_id, batch_img_ids[i])
                mapped_objects = self.map_coordinates_PETS(shared_reg_objects)

            # if DUMP_DATA:
            #    ref_img = cv2.imread(batch_file_names[i])
            #    for obj in mapped_objects:
            #        cv2.rectangle(ref_img, (obj[2], obj[3]), (obj[4], obj[5]), (255, 0, 0), 2)
            #        cv2.imwrite("temp/{}_1.png".format(img_id), ref_img)

            # create prior using detected boxes
            # prior = self.darknet_randomize(objects, False)
            # prior = np.full(shape=(512, 512, 1), fill_value=114, dtype=np.uint8)
            # print("total obj: {}, shared region objects : {}, mapped obj: {}".format(len(objects),
            #                                                                          len(shared_reg_objects),
            #                                                                          len(mapped_objects)))

            # convert shared reg objects to 512x512
            temp = []
            for obj in mapped_objects:
                xmin, ymin, xmax, ymax = int(obj[2]), int(obj[3]), int(obj[4]), int(obj[5])
                # remove negative coordiantes
                xmin = max(0, xmin)
                ymin = max(0, ymin)
                xmax = max(0, xmax)
                ymax = max(0, ymax)
                if self.img_type_prefix == "PNGImages":  # WILDTRACK
                    xmin = int((xmin / 1920.0) * 512)
                    ymin = int((ymin / 1080.0) * 512)
                    xmax = int((xmax / 1920.0) * 512)
                    ymax = int((ymax / 1080.0) * 512)
                elif self.img_type_prefix == "JPEGImages":  # PETS
                    xmin = int((xmin / 720.0) * 512)
                    ymin = int((ymin / 576.0) * 512)
                    xmax = int((xmax / 720.0) * 512)
                    ymax = int((ymax / 576.0) * 512)
                temp.append([obj[0], obj[1], xmin, ymin, xmax, ymax])
            mapped_objects = temp

            for obj in mapped_objects:
                # if obj[0] == 15:
                xmin, ymin, xmax, ymax = int(obj[2]), int(obj[3]), int(obj[4]), int(obj[5])
                aux_channel[ymin:ymax, xmin:xmax] = 255
            # if DUMP_DATA:
            #     print(batch_img_ids[i])
            # cv2.imwrite("temp/{}.jpg".format(batch_img_ids[i]), batch_img)
            # aux_channel = np.full((512, 512, 1), 255, dtype=np.uint8)
            batch_x_prior.append(aux_channel)
        return np.array(batch_x_prior)

    def get_aux_channel_detected_boxes_micro_study(self, batch_file_names, batch_img_ids):
        """
        create mask using detected bounding boxes from full size images. Method used for micro study rather than
        actual performance
        :return:
        """
        DUMP_DATA = True
        shared_reg_bbox = []  # xmin, ymin, xmax, ymax of shared region (drawn on collaborating cam)
        PRIOR_ICOV_TH = 0.10  # min icov value to select an object
        batch_x_prior = []
        # self.collab_cam = 1
        for i in range(len(batch_file_names)):
            aux_channel = np.full((512, 512, 1), 114, dtype=np.uint8)
            if self.img_type_prefix == "PNGImages":  # WILDTRACK
                img_id = batch_img_ids[i]
                collab_img_id = "C{}_{}.png".format(self.collab_cam, img_id[3:])
                file_name = batch_file_names[i]
                # print(file_name)
                # print(img_id)
                img_name_len = len(img_id) + 4
                file_name = file_name[:-1 * img_name_len]
                # file_name = "{}/{}".format(file_name, self.collab_cam)
                collab_file_path = "{}/{}".format(file_name, collab_img_id)
                shared_reg_bbox = [0, 0, 1280, 1080]  # shared regino coords
                #               print("shared_reg_bbox : {}".format(shared_reg_bbox))
                annot_dir = "../dataset/Wildtrack_dataset/Annotations"

            elif self.img_type_prefix == "JPEGImages":  # PETS dataset
                img_id = batch_img_ids[i]
                collab_img_id = "frame_{}_{}.jpg".format(self.collab_cam, img_id[8:])
                file_name = batch_file_names[i]
                img_name_len = len(img_id) + 4
                file_name = file_name[:-1 * img_name_len]
                # file_name = "{}_{}".format(file_name, self.collab_cam)
                collab_file_path = "{}/{}".format(file_name, collab_img_id)
                # print("{}, {}, {}\n".format(batch_file_names[i], batch_img_ids[i], collab_file_path))
                # shared_reg_bbox = [155, 92, 720, 516]  # in collab cam perspective (cam 7 , 8)
                shared_reg_bbox = [0, 0, 720, 380]  # in collab cam perspective (cam 8, 5)
                # shared_reg_bbox = [21, 100, 571, 493]  # in collab cam perspective (cam 5, 7)
                annot_dir = "../dataset/PETS_org/Annotations"
            # read image file
            # objects = self.get_gt_objects(collab_img_id[:-4], annot_dir)
            objects = self.get_gt_objects_WT(batch_img_ids[i], annot_dir)

            if len(objects) == 0:
                batch_x_prior.append(aux_channel)
                continue
            # if DUMP_DATA:
            #     for obj in objects:
            #         cv2.rectangle(collab_img, (obj[2], obj[3]), (obj[4], obj[5]), (255, 0, 0), 2)
            #         cv2.imwrite("temp/{}".format(collab_img_id), collab_img)

            # map coordinates to reference camera coordinate system
            # select objects in shared region
            shared_reg_objects = []
            for obj in objects:
                obj_bbox = [obj[2], obj[3], obj[4], obj[5]]
                icov = self.bb_icov(obj_bbox, shared_reg_bbox)
                if icov >= PRIOR_ICOV_TH:
                    shared_reg_objects.append(obj)

            if len(shared_reg_objects) == 0:
                batch_x_prior.append(aux_channel)
                continue

            # if DUMP_DATA:
            #    ref_img = cv2.imread(batch_file_names[i])
            #    for obj in mapped_objects:
            #        cv2.rectangle(ref_img, (obj[2], obj[3]), (obj[4], obj[5]), (255, 0, 0), 2)
            #        cv2.imwrite("temp/{}_1.png".format(img_id), ref_img)

            # convert shared reg objects to 512x512
            temp = []
            for obj in shared_reg_objects:
                xmin, ymin, xmax, ymax = int(obj[2]), int(obj[3]), int(obj[4]), int(obj[5])
                # remove negative coordiantes
                xmin = max(0, xmin)
                ymin = max(0, ymin)
                xmax = max(0, xmax)
                ymax = max(0, ymax)
                if self.img_type_prefix == "PNGImages":  # WILDTRACK
                    img_id = batch_img_ids[i]
                    xmin = int((xmin / 1920.0) * 512)
                    ymin = int((ymin / 1080.0) * 512)
                    xmax = int((xmax / 1920.0) * 512)
                    ymax = int((ymax / 1080.0) * 512)
                elif self.img_type_prefix == "JPEGImages":  # PETS dataset
                    # img_id = batch_img_ids[i]
                    xmin = int((xmin / 720.0) * 512)
                    ymin = int((ymin / 576.0) * 512)
                    xmax = int((xmax / 720.0) * 512)
                    ymax = int((ymax / 576.0) * 512)
                temp.append([obj[0], obj[1], xmin, ymin, xmax, ymax])
            mapped_objects = temp

            # open file for dumping prior attributes like width, height of bbox
            # file_name = "prior_attr_{}%_{}_micro_study.txt"
            # prior_attr_file = open()
            for obj in mapped_objects:
                # if obj[0] == 15:
                xmin, ymin, xmax, ymax = int(obj[2]), int(obj[3]), int(obj[4]), int(obj[5])

                aux_channel[ymin:ymax, xmin:xmax] = 255
                # if DUMP_DATA:
            #             batch_img = cv2.imread(batch_file_names[i])
            #             assert batch_img is not None
            #             cv2.rectangle(batch_img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
            batch_x_prior.append(aux_channel)
        return np.array(batch_x_prior)

    def get_aux_channel_detected_boxes_cropped_images(self, batch_file_names, batch_img_ids):
        """
        create mask using detected bounding boxes
        :return:
        """
        DUMP_DATA = False
        shared_reg_bbox = []  # xmin, ymin, xmax, ymax of shared region (drawn on collaborating cam)
        ICOV_TH = 0.10  # min icov value to select an object
        batch_x_prior = []
        collab_img_res = (160, 160)  # resolution of collaborating cam image
        print("collab_img_resolution : {}".format(collab_img_res))
        # self.collab_cam = 1
        for i in range(len(batch_file_names)):
            aux_channel = np.full((512, 512, 1), 114, dtype=np.uint8)
            if self.img_type_prefix == "PNGImages":  # WILDTRACK
                img_id = batch_img_ids[i]
                collab_img_id = "C{}_{}.png".format(self.collab_cam, img_id[3:])
                file_name = batch_file_names[i]
                # print(file_name)
                # print(img_id)
                img_name_len = len(img_id) + 4
                file_name = file_name[:-1 * img_name_len]
                # file_name = "{}/{}".format(file_name, self.collab_cam)
                collab_file_path = "{}/{}".format(file_name, collab_img_id)
                # shared_reg_bbox = [1089, 6, 1914, 1071]  # ground truth shared region b/w cam 1,4 (projected on view 4)
                # shared_reg_bbox = [479, 0, 700, 700]  # ground truth shared region b/w cam 1,4 (projected on view 4)
                # shared_reg_bbox = [0, 0, 298, 700]  # ground truth shared region b/w cam 1,4 (projected on view 4)
                shared_reg_bbox = [466, 0, 700, 700]  # gt shared region in 700x700 image
                # print("{}, {}, {}\n".format(batch_file_names[i], batch_img_ids[i], collab_file_path))
                annot_dir = "../dataset/Wildtrack_dataset/Annotations_cropped_700x700"
            elif self.img_type_prefix == "JPEGImages":
                img_id = batch_img_ids[i]
                collab_img_id = "frame_{}_{}.jpg".format(self.collab_cam, img_id[8:])
                file_name = batch_file_names[i]
                img_name_len = len(img_id) + 4
                file_name = file_name[:-1 * img_name_len]
                file_name = "{}_{}".format(file_name[:-3], self.collab_cam)
                collab_file_path = "{}/{}".format(file_name, collab_img_id)
                # print("{}, {}, {}\n".format(batch_file_names[i], batch_img_ids[i], collab_file_path))

            # read image file
            # print(img_id, collab_file_path)
            # print(collab_file_path)
            # collab_img = cv2.imread(collab_file_path)
            # assert collab_img is not None
            # collab_img = cv2.resize(collab_img, dsize=(512, 512), interpolation=cv2.INTER_CUBIC)

            # objects = self.detect_objects(collab_img)
            collab_img = cv2.imread(batch_file_names[i])
            collab_img = cv2.resize(collab_img, dsize=collab_img_res, interpolation=cv2.INTER_CUBIC)
            collab_img = cv2.resize(collab_img, dsize=(700, 700), interpolation=cv2.INTER_CUBIC)
            objects = self.detect_objects(collab_img)
            cv2.imwrite("temp/{}_2.png".format(batch_img_ids[i]), collab_img)
            # objects = self.detect_objects(cv2.imread(batch_file_names[i]))
            # objects = self.get_gt_objects(collab_img_id[:-4])
            # objects = self.get_gt_objects_WT(batch_img_ids[i], annot_dir)
            # if batch_img_ids[i] == "C1_00000035":
            #   print("total objects in gt: {}".format(len(objects))) 
            if len(objects) == 0:
                batch_x_prior.append(aux_channel)
                continue
            # if DUMP_DATA:
            #     for obj in objects:
            #         cv2.rectangle(collab_img, (obj[2], obj[3]), (obj[4], obj[5]), (255, 0, 0), 2)
            #         cv2.imwrite("temp/{}".format(collab_img_id), collab_img)

            # map coordinates to reference camera coordinate system
            # select objects in shared region
            shared_reg_objects = []
            for obj in objects:
                obj_bbox = [obj[2], obj[3], obj[4], obj[5]]
                icov = self.bb_icov(obj_bbox, shared_reg_bbox)
                if icov >= ICOV_TH:
                    shared_reg_objects.append(obj)

            # if batch_img_ids[i] == "C1_00000035":
            #   print("total objects in shared reg: {}".format(len(shared_reg_objects)))
            #   print(shared_reg_objects) 
            # sys.exit(-1)

            if len(shared_reg_objects) == 0:
                batch_x_prior.append(aux_channel)
                continue

            # convert shared reg objects to 512x512 coords
            temp = []
            for obj in shared_reg_objects:
                xmin, ymin, xmax, ymax = int(obj[2]), int(obj[3]), int(obj[4]), int(obj[5])
                xmin = int((xmin / 700.0) * 512)
                ymin = int((ymin / 700.0) * 512)
                xmax = int((xmax / 700.0) * 512)
                ymax = int((ymax / 700.0) * 512)
                temp.append([obj[0], obj[1], xmin, ymin, xmax, ymax])
            shared_reg_objects = temp

            # if batch_img_ids[i] == "C1_00000035":
            #   print("total objects in shared reg: {}".format(len(shared_reg_objects)))
            #   print(shared_reg_objects) 
            # sys.exit(-1)
            # if self.img_type_prefix == "PNGImages":  # WILDTRACK
            #     mapped_objects = self.map_coordinates_WT(shared_reg_objects)
            # elif self.img_type_prefix == "JPEGImages":  # WILDTRACK
            #     # objects = self.map_coordinates_PETS(objects, collab_img_id, batch_img_ids[i])
            #     mapped_objects = self.map_coordinates_PETS(shared_reg_objects)

            # create prior using detected boxes
            # prior = self.darknet_randomize(objects, False)
            # prior = np.full(shape=(512, 512, 1), fill_value=114, dtype=np.uint8)
            # print("total obj: {}, shared region objects : {}, mapped obj: {}".format(len(objects),
            #                                                                          len(shared_reg_objects),
            #                                                                          len(mapped_objects)))
            for obj in shared_reg_objects:
                # if obj[0] == 15:
                xmin, ymin, xmax, ymax = int(obj[2]), int(obj[3]), int(obj[4]), int(obj[5])
                xmin = max(0, xmin)
                ymin = max(0, ymin)
                xmax = max(0, xmax)
                ymax = max(0, ymax)
                aux_channel[ymin:ymax, xmin:xmax] = 255
                # if DUMP_DATA:
            #             batch_img = cv2.imread(batch_file_names[i])
            #             assert batch_img is not None
            #             cv2.rectangle(batch_img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
            # if DUMP_DATA:
            #     print(batch_img_ids[i])
            # cv2.imwrite("temp/{}.jpg".format(batch_img_ids[i]), batch_img)
            # aux_channel = np.full((512, 512, 1), 255, dtype=np.uint8)
            batch_x_prior.append(aux_channel)
        return np.array(batch_x_prior)

    def bb_icov(self, obj_bbox, area_bbox):
        # determine the (x, y)-coordinates of the intersection rectangle
        boxA = obj_bbox
        boxB = area_bbox
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        # boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        # compute the intersection over union by taking the intersection

        icov = interArea / float(boxAArea)
        return np.round(icov, decimals=2)

    def darknet_randomize(self, objects, randomize):
        """
        randomize the ground truth labels and create a prior (mask). Creates different kind of randomization (similar
        to darknet one by Venget).
        :param batch_y_data: transformed labels/annotations
        :return:
        """

        img_height = 512
        img_width = 512

        aux_channel = np.full((img_height, img_width, 1), 114, dtype=np.uint8)

        for index, obj in enumerate(objects):

            # for i, obj in enumerate(img_annot):
            # ignore X% of boxes (no prior for them)
            r_int_1 = np.random.randint(1, 10)
            if r_int_1 > 8:
                continue
            # add remaining objects to the prior
            xmin, ymin, xmax, ymax = obj[2:]
            if randomize:
                r_int = np.random.randint(1, 10)
                if r_int <= 10:
                    # if True:
                    width_rnum = np.random.uniform(-0.20, 0.20)
                    height_rnum = np.random.uniform(-0.20, 0.20)

                    obj_width = xmax - xmin
                    obj_height = ymax - ymin

                    xmin = int(xmin + (obj_width * width_rnum))
                    ymin = int(ymin + (obj_height * height_rnum))
                    xmax = int(xmin + obj_width + (obj_width * width_rnum))
                    ymax = int(ymin + obj_height + (obj_height * height_rnum))

                    # check for out of frame values
                    xmin = max(0, xmin)
                    ymin = max(0, ymin)
                    xmax = min(img_width, xmax)
                    ymax = min(img_height, ymax)

                    aux_channel[ymin:ymax, xmin:xmax] = 255
                # elif r_int == 9:
                #     # white prior (false positive)
                # aux_channel[:, :] = 255
                else:
                    rand_int = random.randint(1, 2)
                    if rand_int == 1:
                        aux_channel[:, :] = 114
                    else:
                        aux_channel[:, :] = 255
            else:
                aux_channel[ymin:ymax, xmin:xmax] = 255
            # batch_y_auxillary.append(aux_channel)

        return aux_channel

    def get_aux_channels_batch(self, batch_X_data, batch_y_data, randomize):
        """
        :param batch_y_data: transformed labels/annotations
        :return:
        """
        img_height = 512
        img_width = 512
        batch_y_auxillary = []

        for index, img_annot in enumerate(batch_y_data):
            aux_channel = np.full((img_height, img_width, 1), 117, dtype=np.uint8)
            for i in range(self.N):
                for obj in img_annot:
                    class_id, xmin, ymin, xmax, ymax = obj
                    if randomize:
                        xmin = xmin + random.randint(-20, 20)
                        ymin = ymin + random.randint(-20, 20)
                        xmax = xmax + random.randint(0, 20)
                        ymax = ymax + random.randint(0, 20)
                        aux_channel[ymin:ymax, xmin:xmax] = 255
                    else:
                        aux_channel[ymin:ymax, xmin:xmax] = 255
            # if self.temp_index < 500:
            #     cv2.imwrite("temp/img_{}.jpg".format(self.temp_index), batch_X_data[index])
            #     cv2.imwrite("temp/img_{}_mask.jpg".format(self.temp_index), aux_channel)
            #     self.temp_index += 1
            # aux_channel[:, :] = 0
            batch_y_auxillary.append(aux_channel)

        return np.array(batch_y_auxillary)

    def dump_raw_data(self, batch_x_data, batch_y_data, batch_x_aux, batch_img_ids):
        # print("dumping data")
        temp_dir = "./temp"
        # out_file = open("{}/batch_y_data.csv".format(temp_dir), 'a')
        # index = random.randint(1, 1000000)
        index = 0
        for bx_data, by_data, bx_aux in zip(batch_x_data, batch_y_data, batch_x_aux):
            # img_file_name = "{}/{}.png".format(temp_dir, index)
            # mask_file_name = "{}/{}_mask.png".format(temp_dir, index)
            img_file_name = "{}/{}.png".format(temp_dir, batch_img_ids[index])
            mask_file_name = "{}/{}_mask.png".format(temp_dir, batch_img_ids[index])

            # draw bounding box on the image
            # print("by_data : {}\n".format(by_data))
            for bbox in by_data:
                # print(bbox)
                if (len(bbox) == 1 or bbox[0] != 15):
                    continue
                cv2.rectangle(bx_data, (bbox[1], bbox[2]), (bbox[3], bbox[4]), (0, 255, 0), 2)
            cv2.imwrite(img_file_name, bx_data)
            cv2.imwrite(mask_file_name, bx_aux)
            # out_file.write("{},{}\n".format(index, len(by_data)))
            index += 1
        # print("finished dumping data")

    def create_mixed_res_imges_PETS(self, batch_x):
        """
        create mixed-resolution images (based on teh collaborating and reference cameras)
        :return:
        """
        SHARED_AREA_RES = (222, 220)  # min possible region of shared region (with same obj det accuracy)
        print("shared area resolution: {}".format(SHARED_AREA_RES))
        # shared_reg_coords = [6, 4, 908, 1074]  # gt overlap cam 1, 4 (projected on view 1)
        # shared_reg_coords = [28, 101, 617, 492] # cam 7, 8
        shared_reg_coords = [0, 0, 720, 576]  # cam 8, 5
        # shared_reg_coords = [91, 142, 693, 510] # cam 5, 7
        print("shared reg coords: {}".format(shared_reg_coords))
        # shared_reg_coords = [0, 0, 298,
        #                     700]  # gt overlap cam 1, 4 (projected on view 1) (for 700x700 img,calculated manually)
        # map shared region to 512x512 (data gen image size)
        xmin_org, ymin_org, xmax_org, ymax_org = shared_reg_coords  # in org cam resolution (720x576 for PETS)
        xmin_tr = int((xmin_org / 720.0) * 512)
        ymin_tr = int((ymin_org / 576.0) * 512)
        xmax_tr = int((xmax_org / 720.0) * 512)
        ymax_tr = int((ymax_org / 576.0) * 512)
        # xmin_tr = int((xmin_org / 700.0) * 512)
        # ymin_tr = int((ymin_org / 700.0) * 512)
        # xmax_tr = int((xmax_org / 700.0) * 512)
        # ymax_tr = int((ymax_org / 700.0) * 512)
        # shared_reg_coords_transformed = [xmin_tr, ymin_tr, xmax_tr, ymax_tr]  # coords in 512x512 image size

        # compute new resolution for shared area
        reg_width = xmax_tr - xmin_tr  # width of shared region in 512x512 image
        reg_height = ymax_tr - ymin_tr
        reg_width_tr = int((reg_width / 512.0) * SHARED_AREA_RES[0])  # new width as per 224x224 overall resolution
        reg_height_tr = int((reg_height / 512.0) * SHARED_AREA_RES[1])
        shared_reg_target_res = (reg_width_tr, reg_height_tr)  # shared area res as per 224x224 overall img res

        batch_x_mixed_res = []
        # modify each image
        for img in batch_x:
            shared_reg = img[ymin_tr:ymax_tr, xmin_tr:xmax_tr]
            temp = cv2.resize(shared_reg, dsize=shared_reg_target_res, interpolation=cv2.INTER_CUBIC)
            shared_reg = cv2.resize(temp, dsize=(reg_width, reg_height), interpolation=cv2.INTER_CUBIC)
            img[ymin_tr:ymax_tr, xmin_tr:xmax_tr] = shared_reg
            batch_x_mixed_res.append(img)
        # assert batch_x.shape == batch_x_mixed_res.shape
        return np.array(batch_x_mixed_res)

    def create_mixed_res_imges_WT(self, batch_x):
        """
        create mixed-resolution images (based on teh collaborating and reference cameras)
        :return:
        """
        SHARED_AREA_RES = (156, 156)  # min possible region of shared region (with same obj det accuracy)
        print("shared area resolution: {}".format(SHARED_AREA_RES))
        # shared_reg_coords = [6, 4, 908, 1074]  # gt overlap cam 1, 4 (projected on view 1)
        # shared_reg_coords = [51, 139, 1507, 1041]  # for camera 5, 7
        shared_reg_coords = [0, 0, 1920, 1080]  # for camera 6, 1
        print("shared reg coords: {}".format(shared_reg_coords))
        # shared_reg_coords = [0, 0, 298,
        #                     700]  # gt overlap cam 1, 4 (projected on view 1) (for 700x700 img,calculated manually)
        # map shared region to 512x512 (data gen image size)
        xmin_org, ymin_org, xmax_org, ymax_org = shared_reg_coords  # in org cam resolution (1920x1080 for WT)
        xmin_tr = int((xmin_org / 1920.0) * 512)
        ymin_tr = int((ymin_org / 1080.0) * 512)
        xmax_tr = int((xmax_org / 1920.0) * 512)
        ymax_tr = int((ymax_org / 1080.0) * 512)
        # xmin_tr = int((xmin_org / 700.0) * 512)
        # ymin_tr = int((ymin_org / 700.0) * 512)
        # xmax_tr = int((xmax_org / 700.0) * 512)
        # ymax_tr = int((ymax_org / 700.0) * 512)
        # shared_reg_coords_transformed = [xmin_tr, ymin_tr, xmax_tr, ymax_tr]  # coords in 512x512 image size

        # compute new resolution for shared area
        reg_width = xmax_tr - xmin_tr  # width of shared region in 512x512 image
        reg_height = ymax_tr - ymin_tr
        reg_width_tr = int((reg_width / 512.0) * SHARED_AREA_RES[0])  # new width as per 224x224 overall resolution
        reg_height_tr = int((reg_height / 512.0) * SHARED_AREA_RES[1])
        shared_reg_target_res = (reg_width_tr, reg_height_tr)  # shared area res as per 224x224 overall img res

        batch_x_mixed_res = []
        # modify each image
        for img in batch_x:
            shared_reg = img[ymin_tr:ymax_tr, xmin_tr:xmax_tr]
            temp = cv2.resize(shared_reg, dsize=shared_reg_target_res, interpolation=cv2.INTER_CUBIC)
            shared_reg = cv2.resize(temp, dsize=(reg_width, reg_height), interpolation=cv2.INTER_CUBIC)
            img[ymin_tr:ymax_tr, xmin_tr:xmax_tr] = shared_reg
            batch_x_mixed_res.append(img)
        # assert batch_x.shape == batch_x_mixed_res.shape
        return np.array(batch_x_mixed_res)

    def extract_shared_region_PETS(self, batch_x, batch_labels, t_batch_indices):
        """
        extract shared rgion from given batch of images and adjust gt labels accordingly
        :return:
        """
        ICOV_TH = 0.75
        MODIFY_LABELS = True  # modify orginal labels (for micro study 2 )
        SHARED_AREA_RES = (224, 224)  # min possible region of shared region (with same obj det accuracy)
        print("shared area resolution: {}".format(SHARED_AREA_RES))
        # shared_reg_coords = [51, 139, 1507, 1041]  # for camera 5, 7
        shared_reg_coords = [0, 0, 720, 380]  # for camera 6, 1
        print("shared reg coords: {}".format(shared_reg_coords))
        batch_labels = deepcopy(self.labels[t_batch_indices[0]:t_batch_indices[1]])

        # map shared region to 512x512 (data gen image size)
        xmin_org, ymin_org, xmax_org, ymax_org = shared_reg_coords  # in org cam resolution (1920x1080 for WT)
        xmin_tr = int((xmin_org / 720.0) * 512)
        ymin_tr = int((ymin_org / 576.0) * 512)
        xmax_tr = int((xmax_org / 720.0) * 512)
        ymax_tr = int((ymax_org / 576.0) * 512)

        # compute new resolution for shared area
        reg_width = xmax_tr - xmin_tr  # width of shared region in 512x512 image
        reg_height = ymax_tr - ymin_tr
        reg_width_tr = int((reg_width / 512.0) * SHARED_AREA_RES[0])  # new width as per 224x224 overall resolution
        reg_height_tr = int((reg_height / 512.0) * SHARED_AREA_RES[1])
        shared_reg_target_res = (reg_width_tr, reg_height_tr)  # shared area res as per 224x224 overall img res

        batch_x_mixed_res = []
        batch_modified_labels = []
        # modify each image
        for img, img_labels in zip(batch_x, batch_labels):
            mixed_res_img = np.full((512, 512, 3), fill_value=114, dtype=np.uint8)
            shared_reg = img[ymin_tr:ymax_tr, xmin_tr:xmax_tr]
            temp = cv2.resize(shared_reg, dsize=shared_reg_target_res, interpolation=cv2.INTER_CUBIC)
            shared_reg = cv2.resize(temp, dsize=(reg_width, reg_height), interpolation=cv2.INTER_CUBIC)
            # img[ymin_tr:ymax_tr, xmin_tr:xmax_tr] = shared_reg
            mixed_res_img[ymin_tr:ymax_tr, xmin_tr:xmax_tr] = shared_reg
            batch_x_mixed_res.append(mixed_res_img)

            # adjust ground truth labels accordingly
            img_modified_lables = []
            for obj in img_labels:
                obj_bbox = [obj[1], obj[2], obj[3], obj[4]]
                icov = self.bb_icov(obj_bbox, shared_reg_coords)
                if icov >= ICOV_TH:
                    # convert labels to 512x512 coordinates
                    # obj[1] = int((obj[1] / 720.0) * 512)
                    # obj[2] = int((obj[2] / 576.0) * 512)
                    # obj[3] = int((obj[3] / 720.0) * 512)
                    # obj[4] = int((obj[4] / 576.0) * 512)
                    temp = [obj[0], obj[1], obj[2], obj[3], obj[4]]
                    img_modified_lables.append(temp)
                else:
                    img_modified_lables.append([18, 0, 0, 720, 576])
            batch_modified_labels.append(img_modified_lables)
        if MODIFY_LABELS:
            self.labels[t_batch_indices[0]:t_batch_indices[1]] = batch_modified_labels
        return np.array(batch_x_mixed_res), batch_modified_labels

    def extract_shared_region_WT(self, batch_x, batch_labels, t_batch_indices):
        """
        extract shared rgion from given batch of images and adjust gt labels accordingly
        :return:
        """
        ICOV_TH = 0.75
        MODIFY_LABELS = True  # modify orginal labels (for micro study 2 )
        SHARED_AREA_RES = (160, 160)  # min possible region of shared region (with same obj det accuracy)
        print("shared area resolution: {}".format(SHARED_AREA_RES))
        # shared_reg_coords = [51, 139, 1507, 1041]  # for camera 5, 7
        shared_reg_coords = [0, 0, 1280, 1080]  # for camera 6, 1

        print("shared reg coords: {}".format(shared_reg_coords))
        batch_labels = deepcopy(self.labels[t_batch_indices[0]:t_batch_indices[1]])
        # map shared region to 512x512 (data gen image size)
        xmin_org, ymin_org, xmax_org, ymax_org = shared_reg_coords  # in org cam resolution (1920x1080 for WT)
        xmin_tr = int((xmin_org / 1920.0) * 512)
        ymin_tr = int((ymin_org / 1080.0) * 512)
        xmax_tr = int((xmax_org / 1920.0) * 512)
        ymax_tr = int((ymax_org / 1080.0) * 512)

        # compute new resolution for shared area
        reg_width = xmax_tr - xmin_tr  # width of shared region in 512x512 image
        reg_height = ymax_tr - ymin_tr
        reg_width_tr = int((reg_width / 512.0) * SHARED_AREA_RES[0])  # new width as per 224x224 overall resolution
        reg_height_tr = int((reg_height / 512.0) * SHARED_AREA_RES[1])
        shared_reg_target_res = (reg_width_tr, reg_height_tr)  # shared area res as per 224x224 overall img res

        batch_x_mixed_res = []
        batch_modified_labels = []
        # modify each image
        for img, img_labels in zip(batch_x, batch_labels):
            mixed_res_img = np.full((512, 512, 3), fill_value=114, dtype=np.uint8)
            shared_reg = img[ymin_tr:ymax_tr, xmin_tr:xmax_tr]
            temp = cv2.resize(shared_reg, dsize=shared_reg_target_res, interpolation=cv2.INTER_CUBIC)
            shared_reg = cv2.resize(temp, dsize=(reg_width, reg_height), interpolation=cv2.INTER_CUBIC)
            # img[ymin_tr:ymax_tr, xmin_tr:xmax_tr] = shared_reg
            mixed_res_img[ymin_tr:ymax_tr, xmin_tr:xmax_tr] = shared_reg
            batch_x_mixed_res.append(mixed_res_img)

            # adjust ground truth labels accordingly
            img_modified_lables = []
            for obj in img_labels:
                obj_bbox = [obj[1], obj[2], obj[3], obj[4]]
                icov = self.bb_icov(obj_bbox, shared_reg_coords)
                if icov >= ICOV_TH:
                    # convert labels to 512x512 coordinates
                    # obj[1] = int((obj[1] / 1920.0) * 512)
                    # obj[2] = int((obj[2] / 1080.0) * 512)
                    # obj[3] = int((obj[3] / 1920.0) * 512)
                    # obj[4] = int((obj[4] / 1080.0) * 512)
                    temp = [obj[0], obj[1], obj[2], obj[3], obj[4]]
                    img_modified_lables.append(temp)
                else:
                    img_modified_lables.append([18, 0, 0, 1920, 1080])
            batch_modified_labels.append(img_modified_lables)
        # replace orginal labels with modeified labels
        if MODIFY_LABELS:
            self.labels[t_batch_indices[0]:t_batch_indices[1]] = batch_modified_labels
        return np.array(batch_x_mixed_res), batch_modified_labels

    def detect_objects(self, img):
        """
        method to detect bounding boxes from using a specific DNN model
        :param input_img:
        :return:
        """
        org_h, org_w, _ = img.shape
        img = cv2.resize(img, dsize=(512, 512), interpolation=cv2.INTER_CUBIC)
        input_images = [img]
        input_images = np.array(input_images)

        y_pred = self.single_img_model.predict(input_images)

        confidence_threshold = 0.5
        y_pred_thresh = [y_pred[k][y_pred[k, :, 1] > confidence_threshold] for k in range(y_pred.shape[0])]
        np.set_printoptions(precision=2, suppress=True, linewidth=90)
        y_pred_decoded = y_pred_thresh
        y_pred_decoded = y_pred_decoded[0].tolist()

        # scle object coordinates back to org image size
        for obj in y_pred_decoded:
            obj[2] = int(obj[2] * (org_w / 512.0))
            obj[3] = int(obj[3] * (org_h / 512.0))
            obj[4] = int(obj[4] * (org_w / 512.0))
            obj[5] = int(obj[5] * (org_h / 512.0))
        return y_pred_decoded

    def generate(self,
                 batch_size=32,
                 shuffle=True,
                 transformations=[],
                 label_encoder=None,
                 returns={'processed_images', 'encoded_labels'},
                 keep_images_without_gt=False,
                 degenerate_box_handling='remove'):
        '''
        Generates batches of samples and (optionally) corresponding labels indefinitely.

        Can shuffle the samples consistently after each complete pass.

        Optionally takes a list of arbitrary image transformations to apply to the
        samples ad hoc.

        Arguments:
            batch_size (int, optional): The size of the batches to be generated.
            shuffle (bool, optional): Whether or not to shuffle the dataset before each pass.
                This option should always be `True` during training, but it can be useful to turn shuffling off
                for debugging or if you're using the generator for prediction.
            transformations (list, optional): A list of transformations that will be applied to the images and labels
                in the given order. Each transformation is a callable that takes as input an image (as a Numpy array)
                and optionally labels (also as a Numpy array) and returns an image and optionally labels in the same
                format.
            label_encoder (callable, optional): Only relevant if labels are given. A callable that takes as input the
                labels of a batch (as a list of Numpy arrays) and returns some structure that represents those labels.
                The general use case for this is to convert labels from their input format to a format that a given object
                detection model needs as its training targets.
            returns (set, optional): A set of strings that determines what outputs the generator yields. The generator's output
                is always a tuple that contains the outputs specified in this set and only those. If an output is not available,
                it will be `None`. The output tuple can contain the following outputs according to the specified keyword strings:
                * 'processed_images': An array containing the processed images. Will always be in the outputs, so it doesn't
                    matter whether or not you include this keyword in the set.
                * 'encoded_labels': The encoded labels tensor. Will always be in the outputs if a label encoder is given,
                    so it doesn't matter whether or not you include this keyword in the set if you pass a label encoder.
                * 'matched_anchors': Only available if `labels_encoder` is an `SSDInputEncoder` object. The same as 'encoded_labels',
                    but containing anchor box coordinates for all matched anchor boxes instead of ground truth coordinates.
                    This can be useful to visualize what anchor boxes are being matched to each ground truth box. Only available
                    in training mode.
                * 'processed_labels': The processed, but not yet encoded labels. This is a list that contains for each
                    batch image a Numpy array with all ground truth boxes for that image. Only available if ground truth is available.
                * 'filenames': A list containing the file names (full paths) of the images in the batch.
                * 'image_ids': A list containing the integer IDs of the images in the batch. Only available if there
                    are image IDs available.
                * 'evaluation-neutral': A nested list of lists of booleans. Each list contains `True` or `False` for every ground truth
                    bounding box of the respective image depending on whether that bounding box is supposed to be evaluation-neutral (`True`)
                    or not (`False`). May return `None` if there exists no such concept for a given dataset. An example for
                    evaluation-neutrality are the ground truth boxes annotated as "difficult" in the Pascal VOC datasets, which are
                    usually treated to be neutral in a model evaluation.
                * 'inverse_transform': A nested list that contains a list of "inverter" functions for each item in the batch.
                    These inverter functions take (predicted) labels for an image as input and apply the inverse of the transformations
                    that were applied to the original image to them. This makes it possible to let the model make predictions on a
                    transformed image and then convert these predictions back to the original image. This is mostly relevant for
                    evaluation: If you want to evaluate your model on a dataset with varying image sizes, then you are forced to
                    transform the images somehow (e.g. by resizing or cropping) to make them all the same size. Your model will then
                    predict boxes for those transformed images, but for the evaluation you will need predictions with respect to the
                    original images, not with respect to the transformed images. This means you will have to transform the predicted
                    box coordinates back to the original image sizes. Note that for each image, the inverter functions for that
                    image need to be applied in the order in which they are given in the respective list for that image.
                * 'original_images': A list containing the original images in the batch before any processing.
                * 'original_labels': A list containing the original ground truth boxes for the images in this batch before any
                    processing. Only available if ground truth is available.
                The order of the outputs in the tuple is the order of the list above. If `returns` contains a keyword for an
                output that is unavailable, that output omitted in the yielded tuples and a warning will be raised.
            keep_images_without_gt (bool, optional): If `False`, images for which there aren't any ground truth boxes before
                any transformations have been applied will be removed from the batch. If `True`, such images will be kept
                in the batch.
            degenerate_box_handling (str, optional): How to handle degenerate boxes, which are boxes that have `xmax <= xmin` and/or
                `ymax <= ymin`. Degenerate boxes can sometimes be in the dataset, or non-degenerate boxes can become degenerate
                after they were processed by transformations. Note that the generator checks for degenerate boxes after all
                transformations have been applied (if any), but before the labels were passed to the `label_encoder` (if one was given).
                Can be one of 'warn' or 'remove'. If 'warn', the generator will merely print a warning to let you know that there
                are degenerate boxes in a batch. If 'remove', the generator will remove degenerate boxes from the batch silently.

        Yields:
            The next batch as a tuple of items as defined by the `returns` argument.
        '''

        if self.dataset_size == 0:
            raise DatasetError("Cannot generate batches because you did not load a dataset.")

        #############################################################################################
        # Warn if any of the set returns aren't possible.
        #############################################################################################

        if self.labels is None:
            if any([ret in returns for ret in
                    ['original_labels', 'processed_labels', 'encoded_labels', 'matched_anchors',
                     'evaluation-neutral']]):
                warnings.warn(
                    "Since no labels were given, none of 'original_labels', 'processed_labels', 'evaluation-neutral', 'encoded_labels', and 'matched_anchors' " +
                    "are possible returns, but you set `returns = {}`. The impossible returns will be `None`.".format(
                        returns))
        elif label_encoder is None:
            if any([ret in returns for ret in ['encoded_labels', 'matched_anchors']]):
                warnings.warn(
                    "Since no label encoder was given, 'encoded_labels' and 'matched_anchors' aren't possible returns, " +
                    "but you set `returns = {}`. The impossible returns will be `None`.".format(returns))
        elif not isinstance(label_encoder, SSDInputEncoder):
            if 'matched_anchors' in returns:
                warnings.warn(
                    "`label_encoder` is not an `SSDInputEncoder` object, therefore 'matched_anchors' is not a possible return, " +
                    "but you set `returns = {}`. The impossible returns will be `None`.".format(returns))

        #############################################################################################
        # Do a few preparatory things like maybe shuffling the dataset initially.
        #############################################################################################

        if shuffle:
            objects_to_shuffle = [self.dataset_indices]
            if not (self.filenames is None):
                objects_to_shuffle.append(self.filenames)
            if not (self.labels is None):
                objects_to_shuffle.append(self.labels)
            if not (self.image_ids is None):
                objects_to_shuffle.append(self.image_ids)
            if not (self.eval_neutral is None):
                objects_to_shuffle.append(self.eval_neutral)
            shuffled_objects = sklearn.utils.shuffle(*objects_to_shuffle)
            for i in range(len(objects_to_shuffle)):
                objects_to_shuffle[i][:] = shuffled_objects[i]

        if degenerate_box_handling == 'remove':
            box_filter = BoxFilter(check_overlap=False,
                                   check_min_area=False,
                                   check_degenerate=True,
                                   labels_format=self.labels_format)

        # Override the labels formats of all the transformations to make sure they are set correctly.
        if not (self.labels is None):
            for transform in transformations:
                transform.labels_format = self.labels_format

        #############################################################################################
        # Generate mini batches.
        #############################################################################################

        current = 0

        while True:

            batch_X, batch_X_aux, batch_y = [], [], []

            if current >= self.dataset_size:
                current = 0

                #########################################################################################
                # Maybe shuffle the dataset if a full pass over the dataset has finished.
                #########################################################################################

                if shuffle:
                    objects_to_shuffle = [self.dataset_indices]
                    if not (self.filenames is None):
                        objects_to_shuffle.append(self.filenames)
                    if not (self.labels is None):
                        objects_to_shuffle.append(self.labels)
                    if not (self.image_ids is None):
                        objects_to_shuffle.append(self.image_ids)
                    if not (self.eval_neutral is None):
                        objects_to_shuffle.append(self.eval_neutral)
                    shuffled_objects = sklearn.utils.shuffle(*objects_to_shuffle)
                    for i in range(len(objects_to_shuffle)):
                        objects_to_shuffle[i][:] = shuffled_objects[i]

            #########################################################################################
            # Get the images, (maybe) image IDs, (maybe) labels, etc. for this batch.
            #########################################################################################

            # We prioritize our options in the following order:
            # 1) If we have the images already loaded in memory, get them from there.
            # 2) Else, if we have an HDF5 dataset, get the images from there.
            # 3) Else, if we have neither of the above, we'll have to load the individual image
            #    files from disk.
            batch_indices = self.dataset_indices[current:current + batch_size]
            if not (self.images is None):
                for i in batch_indices:
                    batch_X.append(self.images[i])
                if not (self.filenames is None):
                    batch_filenames = self.filenames[current:current + batch_size]
                else:
                    batch_filenames = None
            elif not (self.hdf5_dataset is None):
                for i in batch_indices:
                    batch_X.append(self.hdf5_dataset['images'][i].reshape(self.hdf5_dataset['image_shapes'][i]))
                if not (self.filenames is None):
                    batch_filenames = self.filenames[current:current + batch_size]
                else:
                    batch_filenames = None
            else:
                batch_filenames = self.filenames[current:current + batch_size]
                for filename in batch_filenames:
                    with Image.open(filename) as image:
                        batch_X.append(np.array(image, dtype=np.uint8))

            # Get the labels for this batch (if there are any).
            if not (self.labels is None):
                batch_y = deepcopy(self.labels[current:current + batch_size])
                # print(type(batch_y))

                # sys.exit(-1)
            else:
                batch_y = None

            #############################################################################################
            # amit : store the indices of batch
            t_batch_indices = [current, current + batch_size]  # temp variable
            #############################################################################################

            if not (self.eval_neutral is None):
                batch_eval_neutral = self.eval_neutral[current:current + batch_size]
            else:
                batch_eval_neutral = None

            # Get the image IDs for this batch (if there are any).
            if not (self.image_ids is None):
                batch_image_ids = self.image_ids[current:current + batch_size]
            else:
                batch_image_ids = None

            if 'original_images' in returns:
                batch_original_images = deepcopy(batch_X)  # The original, unaltered images
            if 'original_labels' in returns:
                batch_original_labels = deepcopy(batch_y)  # The original, unaltered labels

            current += batch_size

            #########################################################################################
            # Maybe perform image transformations.
            #########################################################################################

            batch_items_to_remove = []  # In case we need to remove any images from the batch, store their indices in this list.
            batch_inverse_transforms = []

            for i in range(len(batch_X)):

                if not (self.labels is None):
                    # Convert the labels for this image to an array (in case they aren't already).
                    batch_y[i] = np.array(batch_y[i])
                    # If this image has no ground truth boxes, maybe we don't want to keep it in the batch.
                    if (batch_y[i].size == 0) and not keep_images_without_gt:
                        batch_items_to_remove.append(i)
                        batch_inverse_transforms.append([])
                        continue

                # Apply any image transformations we may have received.
                if transformations:

                    inverse_transforms = []

                    for transform in transformations:

                        if not (self.labels is None):

                            if ('inverse_transform' in returns) and (
                                    'return_inverter' in inspect.signature(transform).parameters):
                                batch_X[i], batch_y[i], inverse_transform = transform(batch_X[i], batch_y[i],
                                                                                      return_inverter=True)
                                inverse_transforms.append(inverse_transform)
                            else:
                                batch_X[i], batch_y[i] = transform(batch_X[i], batch_y[i])

                            if batch_X[
                                i] is None:  # In case the transform failed to produce an output image, which is possible for some random transforms.
                                batch_items_to_remove.append(i)
                                batch_inverse_transforms.append([])
                                continue

                        else:

                            if ('inverse_transform' in returns) and (
                                    'return_inverter' in inspect.signature(transform).parameters):
                                batch_X[i], inverse_transform = transform(batch_X[i], return_inverter=True)
                                inverse_transforms.append(inverse_transform)
                            else:
                                batch_X[i] = transform(batch_X[i])

                    batch_inverse_transforms.append(inverse_transforms[::-1])

                #########################################################################################
                # Check for degenerate boxes in this batch item.
                #########################################################################################

                if not (self.labels is None):

                    xmin = self.labels_format['xmin']
                    ymin = self.labels_format['ymin']
                    xmax = self.labels_format['xmax']
                    ymax = self.labels_format['ymax']

                    if np.any(batch_y[i][:, xmax] - batch_y[i][:, xmin] <= 0) or np.any(
                            batch_y[i][:, ymax] - batch_y[i][:, ymin] <= 0):
                        if degenerate_box_handling == 'warn':
                            warnings.warn(
                                "Detected degenerate ground truth bounding boxes for batch item {} with bounding boxes {}, ".format(
                                    i, batch_y[i]) +
                                "i.e. bounding boxes where xmax <= xmin and/or ymax <= ymin. " +
                                "This could mean that your dataset contains degenerate ground truth boxes, or that any image transformations you may apply might " +
                                "result in degenerate ground truth boxes, or that you are parsing the ground truth in the wrong coordinate format." +
                                "Degenerate ground truth bounding boxes may lead to NaN errors during the training.")
                        elif degenerate_box_handling == 'remove':
                            batch_y[i] = box_filter(batch_y[i])
                            if (batch_y[i].size == 0) and not keep_images_without_gt:
                                batch_items_to_remove.append(i)

            #########################################################################################
            # Remove any items we might not want to keep from the batch.
            #########################################################################################

            if batch_items_to_remove:
                for j in sorted(batch_items_to_remove, reverse=True):
                    # This isn't efficient, but it hopefully shouldn't need to be done often anyway.
                    batch_X.pop(j)
                    batch_filenames.pop(j)
                    if batch_inverse_transforms: batch_inverse_transforms.pop(j)
                    if not (self.labels is None): batch_y.pop(j)
                    if not (self.image_ids is None): batch_image_ids.pop(j)
                    if not (self.eval_neutral is None): batch_eval_neutral.pop(j)
                    if 'original_images' in returns: batch_original_images.pop(j)
                    if 'original_labels' in returns and not (self.labels is None): batch_original_labels.pop(j)

            #########################################################################################

            # CAUTION: Converting `batch_X` into an array will result in an empty batch if the images have varying sizes
            #          or varying numbers of channels. At this point, all images must have the same size and the same
            #          number of channels.

            # if self.test_resolution != (300, 300):
            #     # print("going to change resolution..")
            #     batch_X = self.apply_multi_resolutions_transform(batch_images=batch_X)

            batch_X = np.array(batch_X)
            # print("shape of batch_X : {}".format(batch_X.shape))

            if (batch_X.size == 0):
                raise DegenerateBatchError(
                    "You produced an empty batch. This might be because the images in the batch vary " +
                    "in their size and/or number of channels. Note that after all transformations " +
                    "(if any were given) have been applied to all images in the batch, all images " +
                    "must be homogenous in size along all axes.")

            #########################################################################################
            # If we have a label encoder, encode our labels.
            #########################################################################################

            if not (label_encoder is None or self.labels is None):

                if ('matched_anchors' in returns) and isinstance(label_encoder, SSDInputEncoder):
                    batch_y_encoded, batch_matched_anchors = label_encoder(batch_y, diagnostics=True)
                else:
                    batch_y_encoded = label_encoder(batch_y, diagnostics=False)
                    batch_matched_anchors = None

            else:
                batch_y_encoded = None
                batch_matched_anchors = None

            # ################################ Generate Priors ####################################################

            # batch_X_aux = self.get_aux_channels_batch(batch_X_data=batch_X, batch_y_data=batch_y, randomize=True)
            # batch_X_aux = self.get_aux_channels_batch_darknet_randomization(batch_X_data=batch_X, batch_y_data=batch_y,
            #                                                                 randomize=True)
            batch_X_aux = self.get_aux_channel_detected_boxes(batch_filenames, batch_image_ids)
            # batch_X_aux = self.get_aux_channel_detected_boxes_micro_study(batch_filenames, batch_image_ids)
            # batch_X_aux = self.get_aux_channel_detected_boxes_cropped_images(batch_filenames, batch_image_ids)

            # create mixed resolution images in the batch
            if self.test_dataset == "WILDTRACK":
                batch_X = self.create_mixed_res_imges_WT(batch_X)
            elif self.test_dataset == "PETS":
                batch_X = self.create_mixed_res_imges_PETS(batch_X)
            else:
                print("WRONG Dataset")
                sys.exit(-1)

            # ############################### Compare shared regions Avg Prec Scores ##########################
            # print("Returns: {}\n".format(returns))

            # if self.test_dataset == "WILDTRACK":
            #     batch_X, batch_original_labels = self.extract_shared_region_WT(batch_X, batch_y, t_batch_indices)
            # elif self.test_dataset == "PETS":
            #     batch_X, batch_original_labels = self.extract_shared_region_PETS(batch_X, batch_y, t_batch_indices)
            # else:
            #     print("WRONG Dataset")
            #     sys.exit(-1)
            # #################################################################################################

            # batch_X = self.apply_resolution_transform(batch_X)

            t_batch_y = deepcopy(self.labels[t_batch_indices[0]:t_batch_indices[1]])
            self.dump_raw_data(batch_x_data=deepcopy(batch_X), batch_y_data=t_batch_y, batch_x_aux=batch_X_aux,
                               batch_img_ids=batch_image_ids)
            # self.dump_raw_data(batch_x_data=batch_X.copy(), batch_y_data=batch_original_labels, batch_x_aux=batch_X_aux,
            #                   batch_img_ids=batch_image_ids)
            # print("shape of batch_x_aux : {}".format(batch_X_aux.shape))
            ########################################################################################

            #########################################################################################
            # Compose the output.
            #########################################################################################

            ret = []
            if 'processed_images' in returns: ret.append([batch_X, batch_X_aux])
            if 'encoded_labels' in returns: ret.append(batch_y_encoded)
            if 'matched_anchors' in returns: ret.append(batch_matched_anchors)
            if 'processed_labels' in returns: ret.append(batch_y)
            if 'filenames' in returns: ret.append(batch_filenames)
            if 'image_ids' in returns: ret.append(batch_image_ids)
            if 'evaluation-neutral' in returns: ret.append(batch_eval_neutral)
            if 'inverse_transform' in returns: ret.append(batch_inverse_transforms)
            if 'original_images' in returns: ret.append(batch_original_images)
            if 'original_labels' in returns: ret.append(batch_original_labels)

            yield ret

    def apply_resolution_transform(self, batch_x):
        batch_x_transformed = []
        batch_y_transformed = []
        # transform each image and labels

        for img in batch_x:
            # img_t, label_t = scale(image=img, labels=label)
            # img_t = cv2.cvtColor(img_t, code=cv2.COLOR_BGR2RGB)
            # batch_x_transformed.append(img_t)
            # batch_y_transformed.append(label_t)
            img_t = cv2.resize(img, dsize=(self.resolution, self.resolution), interpolation=cv2.INTER_CUBIC)
            img_t = cv2.resize(img_t, dsize=(512, 512), interpolation=cv2.INTER_CUBIC)
            batch_x_transformed.append(img_t)
        return np.array(batch_x_transformed)

    def save_dataset(self,
                     filenames_path='filenames.pkl',
                     labels_path=None,
                     image_ids_path=None,
                     eval_neutral_path=None):
        '''
        Writes the current `filenames`, `labels`, and `image_ids` lists to the specified files.
        This is particularly useful for large datasets with annotations that are
        parsed from XML files, which can take quite long. If you'll be using the
        same dataset repeatedly, you don't want to have to parse the XML label
        files every time.

        Arguments:
            filenames_path (str): The path under which to save the filenames pickle.
            labels_path (str): The path under which to save the labels pickle.
            image_ids_path (str, optional): The path under which to save the image IDs pickle.
            eval_neutral_path (str, optional): The path under which to save the pickle for
                the evaluation-neutrality annotations.
        '''
        with open(filenames_path, 'wb') as f:
            pickle.dump(self.filenames, f)
        if not labels_path is None:
            with open(labels_path, 'wb') as f:
                pickle.dump(self.labels, f)
        if not image_ids_path is None:
            with open(image_ids_path, 'wb') as f:
                pickle.dump(self.image_ids, f)
        if not eval_neutral_path is None:
            with open(eval_neutral_path, 'wb') as f:
                pickle.dump(self.eval_neutral, f)

    def get_dataset(self):
        '''
        Returns:
            4-tuple containing lists and/or `None` for the filenames, labels, image IDs,
            and evaluation-neutrality annotations.
        '''
        return self.filenames, self.labels, self.image_ids, self.eval_neutral

    def get_dataset_size(self):
        '''
        Returns:
            The number of images in the dataset.
        '''
        return self.dataset_size
