"""
utility module for computing various feature extractors for objects
"""

import cv2
import numpy as np
from src.server_files import dnn_client as dc
import pickle


def compute_color_hist(img_obj):
    channels = [0, 1, 2]
    # bins = [90, 128]
    bins = [45, 64, 64]
    ranges = [0, 180, 0, 256, 0, 256]
    # match_method = cv2.HISTCMP_BHATTACHARYYA

    # calculate color histograms
    # img_obj_1 = cv2.equalizeHist(img_obj_1)
    img_obj_hsv = cv2.cvtColor(img_obj, cv2.COLOR_BGR2HSV)
    color_2d_hist = cv2.calcHist([img_obj_hsv], channels, None, bins, ranges)
    return color_2d_hist


def compute_sift_features(img_obj):
    surf = cv2.xfeatures2d.SURF_create()
    keypoints, descriptions = surf.detectAndCompute(img_obj, None)
    return descriptions


def get_eucl_distance(vector_1, vector_2):
    vector_1 = np.array(vector_1)
    vector_2 = np.array(vector_2)
    eucl_distance = np.sqrt(np.sum(np.square(vector_1 - vector_2)))
    return eucl_distance


def same_objects(vector1, vector2):
    """
    classify both objects and returns if they are same category objects or different
    :param vector1:
    :param vector2:
    :return:
    """
    # vector1 = vector1.reshape(1, -1)
    # vector2 = vector2.reshape(1, -1)
    cat_1 = classifier.predict([vector1])
    cat_2 = classifier.predict([vector2])

    return cat_1[0] == cat_2[0]


def get_obj_embedding(img_obj, temp_dir_path):
    """
    retrieves the object embeddings from the dnnn
    :param img_obj:
    :return:
    """
    # write image to temporary file
    temp_img_path = "{}/temp_image.jpg".format(temp_dir_path)
    cv2.imwrite(temp_img_path, img_obj)

    embedding = dc.get_embeddings(test_img_path=temp_img_path)
    return embedding


def compute_img_moments():
    print("")


def compare_color_hist(obj1, obj2):
    match_method = cv2.HISTCMP_BHATTACHARYYA
    score = cv2.compareHist(obj1['color'], obj2['color'], match_method)
    score = np.around(score, decimals=2)
    return score


def compare_sift(obj1, obj2):
    lowe_ratio_coeff = 0.75
    index_params = dict(algorithm=0, trees=5)
    search_params = dict(checks=50)
    flann_matcher = cv2.FlannBasedMatcher(index_params, search_params)

    if (obj1['sift'] is None) or (obj2['sift'] is None) or len(obj1['sift']) < 2 or len(obj2['sift']) < 2:
        return 0
    matches = flann_matcher.knnMatch(obj1['sift'], obj2['sift'], k=2)
    good_matches_1 = []
    total_matches_1 = len(matches)
    for m1, m2 in matches:
        if m1.distance < lowe_ratio_coeff * m2.distance:
            good_matches_1.append([m1])

    matches = flann_matcher.knnMatch(obj2['sift'], obj1['sift'], k=2)
    good_matches_2 = []
    total_matches_2 = len(matches)
    for m1, m2 in matches:
        if m1.distance < lowe_ratio_coeff * m2.distance:
            good_matches_2.append([m1])

    # select the maximum matches and normalize the score
    if len(good_matches_1) >= len(good_matches_2):
        good_matches_fraction = len(good_matches_1) / total_matches_1
    else:
        good_matches_fraction = len(good_matches_2) / total_matches_2

    return np.around(good_matches_fraction, decimals=2)


def bb_iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return np.round(iou, decimals=2)


# load svm model
model_file = open("svm_model_8_samples(82_82).sav", 'rb')
classifier = pickle.load(model_file)
