"""
features that can be used to compute similarity score between two objects
"""

import cv2
import numpy as np
import math


def compare_hist(img_obj_1, img_obj_2):
    channels = [0, 1, 2]
    # bins = [90, 128]
    bins = [45, 64, 64]
    ranges = [0, 180, 0, 256, 0, 256]
    match_method = cv2.HISTCMP_BHATTACHARYYA

    # calculate color histograms
    # img_obj_1 = cv2.equalizeHist(img_obj_1)
    img_obj_1 = cv2.cvtColor(img_obj_1, cv2.COLOR_BGR2HSV)
    color_2d_hist_1 = cv2.calcHist([img_obj_1], channels, None, bins, ranges)
    img_obj_2 = cv2.cvtColor(img_obj_2, cv2.COLOR_BGR2HSV)
    # img_obj_2 = cv2.equalizeHist(img_obj_2)
    color_2d_hist_2 = cv2.calcHist([img_obj_2], channels, None, bins, ranges)

    # compare histograms
    score = cv2.compareHist(color_2d_hist_1, color_2d_hist_2, match_method)
    score = np.around(score, decimals=2)

    return score


def compare_hu_moments(img_obj_1, img_obj_2):
    img_obj_1 = cv2.cvtColor(img_obj_1, cv2.COLOR_BGR2GRAY)
    # _, img_obj_1 = cv2.threshold(img_obj_1, 128, 255, cv2.THRESH_BINARY)
    # moments = cv2.moments(img_obj_1)
    # # Calculate Hu Moments
    # huMoments_1 = cv2.HuMoments(moments)
    # # Log scale hu moments
    # for i in range(0, 7):
    #     log_10 = math.log(50,base=10)

    img_obj_2 = cv2.cvtColor(img_obj_2, cv2.COLOR_BGR2GRAY)
    # _, img_obj_2 = cv2.threshold(img_obj_2, 128, 255, cv2.THRESH_BINARY)
    # moments = cv2.moments(img_obj_2)
    # # Calculate Hu Moments
    # huMoments_2 = cv2.HuMoments(moments)

    hu_score = cv2.matchShapes(img_obj_1, img_obj_2, cv2.CONTOURS_MATCH_I2, 0)
    return hu_score


def compare_sift(img_obj_1, img_obj_2):
    surf = cv2.xfeatures2d.SURF_create()
    HESSIAN_THRESHOLD = 300
    LOWE_RATIO_TEST_COEFFICIENT = 0.75
    index_params = dict(algorithm=0, trees=5)
    search_params = dict(checks=50)
    flann_matcher = cv2.FlannBasedMatcher(index_params, search_params)

    # calcuate features
    keypoints_1, descriptions_1 = surf.detectAndCompute(img_obj_1, None)
    keypoints_2, descriptions_2 = surf.detectAndCompute(img_obj_2, None)

    # match features
    matches = flann_matcher.knnMatch(descriptions_1, descriptions_2, k=2)
    good_matches_1 = []
    total_matches_1 = len(matches)
    for m1, m2 in matches:
        if m1.distance < LOWE_RATIO_TEST_COEFFICIENT * m2.distance:
            good_matches_1.append([m1])

    matches = flann_matcher.knnMatch(descriptions_2, descriptions_1, k=2)
    good_matches_2 = []
    total_matches_2 = len(matches)
    for m1, m2 in matches:
        if m1.distance < LOWE_RATIO_TEST_COEFFICIENT * m2.distance:
            good_matches_2.append([m1])

    # select the maximum matches and normalize the score
    if len(good_matches_1) >= len(good_matches_2):
        good_matches = len(good_matches_1) / total_matches_1
    else:
        good_matches = len(good_matches_2) / total_matches_2

    return np.around(good_matches, decimals=2)


def test_hist():
    image_1 = "D:\GitRepo\edge_computing\edge_collaboration\ssd_keras\examples\\fish_bike.jpg"
    image_2 = "D:\GitRepo\edge_computing\edge_collaboration\ssd_keras\examples\\trained_ssd300_pascalVOC2007_test_pred_03.png"
    obj_1 = cv2.imread(image_1)
    # cv2.imshow("org", obj_1)
    # cv2.waitKey(-1)
    obj_2 = cv2.imread(image_2)
    score_hist = compare_hist(obj_1, obj_2)
    score_sift = compare_sift(obj_1, obj_2)
    score_hu = compare_hu_moments(obj_1, obj_2)
    print("score_hist : {}, score_sift: {}, score_hu : {}".format(score_hist, score_sift, score_hu))


# test_hist()
