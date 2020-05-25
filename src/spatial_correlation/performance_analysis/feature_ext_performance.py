"""
module to quantify accuracy of object comparison features e.g. precision/recall of color-hist based method.
"""

import cv2
import pandas as pd
import utils
import numpy as np
from sklearn.metrics import confusion_matrix
import math
from matplotlib import pyplot as plt
import random
import time

THR_color_hist = 0.2
THR_sift = 0.3
DEBUG = False


def create_frame_name(f_num):
    """
    crearte frame name from frame number
    :param f_num:
    :return:
    """
    if f_num < 10:
        return "frame_000{}".format(f_num)
    elif f_num < 100:
        return "frame_00{}".format(f_num)
    elif f_num < 1000:
        return "frame_0{}".format(f_num)


def get_object_features(img_obj, obj_loc, obj_id):
    """
    extract features using feature extractors and create a dict
    :param img_obj:
    :param obj_id:
    :return:
    """
    # time1 = time.time()
    features = {
        'rand_id': random.randint(10000, 99999),
        'location': obj_loc,
        'track_id': obj_id}
    if INCLUDE_COLOR:
        features['color'] = utils.compute_color_hist(img_obj)
    if INCLUDE_SIFT:
        features['sift'] = utils.compute_sift_features(img_obj)
    if INCLUDE_EMBEDDING:
        features['embedding'] = utils.get_obj_embedding(img_obj, temp_dir_path="../../temp_files")

    # print("time in feature extraction: {}".format(time.time() - time1))
    return features


def compare_objects(obj1, obj2, color_th, sift_threshold):
    objects_same = False
    # color_score = utils.compare_color_hist(obj1, obj2)
    sift_score = utils.compare_sift(obj1, obj2)

    # logic for objects being same
    # if color_score <= THR_color_hist or sift_score >= THR_sift:
    # if color_score <= color_th:
    if sift_score >= sift_threshold:
        objects_same = True
    return objects_same


def get_prec_recall(actual, estimated, labels):
    """
    sklearn confusion matrix is such that left-to-right is predicted labels and top-2-bottom is actual labels.
    The parameter labels is very important, the first label in labels list is the label for which tp, fp etc
    are calculated. Thus precision/recall values are reported for that
    class only and not averaged over all the classes (like in multi-class classification)
    :param actual:
    :param estimated:
    :param labels:
    :return:
    """
    cm = confusion_matrix(y_true=actual, y_pred=estimated, labels=labels)
    tp, fn, fp, tn = cm.ravel()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    # replace nan values with 0.0
    precision = np.nan_to_num(precision, 0.0)
    recall = np.nan_to_num(recall, 0.0)

    return precision, recall


def visualize_obj(obj, frame, win_title):
    x1, y1 = obj['location'][0]
    x2, y2 = obj['location'][1]
    cv2.imshow(win_title, frame[y1:y2, x1:x2])
    cv2.waitKey(10)


def get_scores(obj1, obj2):
    if INCLUDE_COLOR:
        color_score = utils.compare_color_hist(obj1, obj2)
    else:
        color_score = None

    if INCLUDE_SIFT:
        sift_score = utils.compare_sift(obj1, obj2)
    else:
        sift_score = None

    if INCLUDE_EMBEDDING:
        eucl_distance = utils.get_eucl_distance(obj1['embedding'], obj2['embedding'])
    else:
        eucl_distance = None

    return color_score, sift_score, eucl_distance


def match_objects(objects_1, objects_2, th):
    """
    matches objects in one set to the objects in other set, finds best possible match using the compare_objects method
    :param objects_1:
    :param objects_2:
    :return:
    """
    est_equality = []
    actual_equality = []
    for obj1 in objects_1:
        for obj2 in objects_2:
            color_score, sift_score, eucl_distance = get_scores(obj1, obj2)

            if INCLUDE_COLOR:
                total_score = color_score
            elif INCLUDE_SIFT:
                total_score = sift_score
            elif INCLUDE_EMBEDDING:
                total_score = eucl_distance
            # total_score = sift_score
            print("total_score: {}".format(total_score))

            # if total_score >= th:
            if sift_score >= 0.05 or eucl_distance <= 13:
                # objects are same as per the feature used and its threshold
                est_equality.append("same")
            else:
                est_equality.append("not-same")

            if obj1['track_id'] == obj2['track_id']:
                actual_equality.append("same")
            else:
                actual_equality.append("not-same")

    return est_equality, actual_equality


def match_objects_old(objects_1, objects_2, th):
    """
    matches objects in one set to the objects in other set, finds best possible match using the compare_objects method
    :param objects_1:
    :param objects_2:
    :return:
    """
    object_mappings = []

    for obj1 in objects_1:
        # best_score = -1
        best_score = 1000  # high for eucl_distance feature
        matching_obj = -1
        for obj2 in objects_2:
            color_score, sift_score, eucl_distance = get_scores(obj1, obj2)
            # total_score = sift_score + (1 - color_score)
            # total_score = sift_score
            total_score = color_score
            if total_score < best_score:
                matching_obj = obj2
                best_score = total_score
        # if (-1 < best_score >= th) and matching_obj != -1:
        if (best_score <= th) and matching_obj != -1:
            object_mappings.append(
                [obj1['rand_id'], matching_obj['rand_id'], best_score, obj1['location'], matching_obj['location'],
                 obj1['track_id'], matching_obj['track_id']])

    final_obj_mappings_df = pd.DataFrame(columns=['obj_1_id', 'obj_2_id', 'matching_score', 'obj_1_loc',
                                                  'obj_2_loc', 'obj_1_track_id', 'obj_2_track_id'])
    if len(object_mappings) > 0:
        dataframe = pd.DataFrame(object_mappings,
                                 columns=['obj_1_id', 'obj_2_id', 'matching_score', 'obj_1_loc',
                                          'obj_2_loc', 'obj_1_track_id', 'obj_2_track_id'])

        # select the pairs with maximum matching score (in case of 1 to many mappings)
        # final_obj_mappings = []

        df_groups = dataframe.groupby(by=dataframe['obj_2_id'])
        for grp_name, group in df_groups:
            max_score_index = group['matching_score'].idxmax()
            # obj_map = dataframe.iloc[max_score_index].values
            final_obj_mappings_df = final_obj_mappings_df.append(dataframe.iloc[max_score_index])
            # final_obj_mappings.append([int(obj_map[0]), int(obj_map[1]), obj_map[2]])
    # return final_obj_mappings
    return final_obj_mappings_df


def find_common_objects(view_1_objects, view_2_objects):
    """
    using the ground truth, find common objects between both the frames
    :param view_1_objects:
    :param view_2_objects:
    :return:
    """
    view_1_common_objects = pd.DataFrame(columns=['track_id', 'xmin', 'ymin', 'xmax', 'ymax', 'frame_number', 'lost',
                                                  'occluded', 'generated', 'label'])
    view_2_common_objects = pd.DataFrame(columns=['track_id', 'xmin', 'ymin', 'xmax', 'ymax', 'frame_number', 'lost',
                                                  'occluded', 'generated', 'label'])
    common_obj_list = list(set(view_1_objects.track_id) & set(view_2_objects.track_id))
    # print(common_obj_list)
    for item in common_obj_list:
        v_1_obj = view_1_objects[view_1_objects['track_id'] == item]
        view_1_common_objects = view_1_common_objects.append(v_1_obj)

        v_2_obj = view_2_objects[view_2_objects['track_id'] == item]
        view_2_common_objects = view_2_common_objects.append(v_2_obj)

    return view_1_common_objects, view_2_common_objects


def get_all_objects_features(view_n_objects, view_n_frame):
    """

    :param view_n_objects:
    :param view_n_frame:
    :return:
    """
    view_n_obj_features = []
    for _, obj in view_n_objects.iterrows():
        img = view_n_frame[obj['ymin']:obj['ymax'], obj['xmin']:obj['xmax']]
        obj_location = [(obj['xmin'], obj['ymin']), (obj['xmax'], obj['ymax'])]
        obj_track_id = obj['track_id']
        view_n_obj_features.append(get_object_features(img, obj_location, obj_track_id))

    return view_n_obj_features


def find_distances(v_1_objects_features, v_2_objects_features, feature):
    """
    find object wise distance (based on given feature)
    :param v_1_objects_features:
    :param v_2_objects_features:
    :return:
    """
    distances = []
    for i in range(len(v_1_objects_features)):
        obj1_features = v_1_objects_features[i]
        obj2_features = v_2_objects_features[i]
        color_score, sift_score, eucl_distance = get_scores(obj1_features, obj2_features)
        if feature == "color":
            distances.append(color_score)
        elif feature == "sift":
            distances.append(sift_score)
        elif feature == "embedding":
            distances.append(eucl_distance)
        else:
            print("Wrong Feature selected")

    return distances


if __name__ == "__main__":
    SAME_OBJECTS_ANALYSIS = False  # whether to do same obj analysis or not
    INCLUDE_COLOR = False
    INCLUDE_SIFT = True
    INCLUDE_EMBEDDING = True

    dir_name = "../../../dataset/pets_training"
    view_1_name = "View_008"
    view_2_name = "View_007"

    view_1_grnd_truth = pd.read_csv("{}/{}.txt".format(dir_name, view_1_name), delimiter=" ")
    view_2_grnd_truth = pd.read_csv("{}/{}.txt".format(dir_name, view_2_name), delimiter=" ")

    # frame_number = 1
    colorhist_thresholds = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    eucl_distances = [0, 2, 5, 10, 15, 18, 20, 30]
    # eucl_distances = [10]
    precisions = []
    recalls = []

    if INCLUDE_SIFT or INCLUDE_COLOR:
        threshold_list = colorhist_thresholds
    elif INCLUDE_EMBEDDING:
        threshold_list = eucl_distances
    # for th in colorhist_thresholds:
    for th in threshold_list:
        print("threshold : {}".format(th))
        # th = 0.3
        est_obj_equality = []  # "same" : when both objects are estimated to be same, not-same otherwise
        actual_obj_equality = []  # ground truth whether both objects are same or not-same
        distances = []
        for frame_number in range(10, 60):
            print("frame_number : {}".format(frame_number))
            # frame_number = 5
            frame_name = create_frame_name(frame_number)

            view_1_frame = cv2.imread("{}/{}/{}.jpg".format(dir_name, view_1_name, frame_name))
            view_2_frame = cv2.imread("{}/{}/{}.jpg".format(dir_name, view_2_name, frame_name))

            # get objects from frames (using ground truth)
            view_1_objects = view_1_grnd_truth[view_1_grnd_truth['frame_number'] == frame_number]
            view_2_objects = view_2_grnd_truth[view_2_grnd_truth['frame_number'] == frame_number]
            # apply filters on objects
            view_1_objects = view_1_objects[(view_1_objects['lost'] == 0) & (view_1_objects['occluded'] == 0)]
            view_2_objects = view_2_objects[(view_2_objects['lost'] == 0) & (view_2_objects['occluded'] == 0)]

            if SAME_OBJECTS_ANALYSIS:
                # get common objects between view 1 adn view 2
                v_1_comm_objects, v_2_comm_objects = find_common_objects(view_1_objects=view_1_objects,
                                                                         view_2_objects=view_2_objects)
                # get features of common objects
                v_1_comm_objects_features = get_all_objects_features(v_1_comm_objects, view_1_frame)
                v_2_comm_objects_features = get_all_objects_features(v_2_comm_objects, view_2_frame)

                # find similarity (or distance) between objects
                distances = distances + find_distances(v_1_comm_objects_features, v_2_comm_objects_features,
                                                       feature="sift")

            # compute object features for each object in frame
            view_1_obj_features = get_all_objects_features(view_1_objects, view_1_frame)
            view_2_obj_features = get_all_objects_features(view_2_objects, view_2_frame)

            # for _, obj in view_1_objects.iterrows():
            #     img = view_1_frame[obj['ymin']:obj['ymax'], obj['xmin']:obj['xmax']]
            #     obj_location = [(obj['xmin'], obj['ymin']), (obj['xmax'], obj['ymax'])]
            #     obj_track_id = obj['track_id']
            #     view_1_obj_features.append(get_object_features(img, obj_location, obj_track_id))
            #
            # for _, obj in view_2_objects.iterrows():
            #     img = view_2_frame[obj['ymin']:obj['ymax'], obj['xmin']:obj['xmax']]
            #     obj_location = [(obj['xmin'], obj['ymin']), (obj['xmax'], obj['ymax'])]
            #     obj_track_id = obj['track_id']
            #     view_2_obj_features.append(get_object_features(img, obj_location, obj_track_id))

            # compare objects from view 1 with objects from view 2

            # find mapping b/w objects in view 1 to objects in view 2
            # obj_mappings = match_objects(view_1_obj_features, view_2_obj_features, th, est_obj_equality,
            #                              actual_obj_equality)
            est_equality, actual_equality = match_objects(view_1_obj_features, view_2_obj_features, th)
            est_obj_equality = est_obj_equality + est_equality  # list concat
            actual_obj_equality = actual_obj_equality + actual_equality
            # for index, row in obj_mappings.iterrows():
            #     est_obj_equality.append("same")
            #     if row['obj_id_1_track_id'] == row['obj_id_2_track_id']:
            #         actual_obj_equality.append("same")
            #     else:
            #         actual_obj_equality.append("not-same")

            # for obj1 in view_1_obj_features:
            #     for obj2 in view_2_obj_features:
            #         d = obj_mappings[(obj_mappings['obj_1_id'] == obj1['rand_id']) &
            #                          (obj_mappings['obj_2_id'] == obj2['rand_id'])]
            #         # estimated object equality
            #         if len(d) > 0:
            #             est_obj_equality.append("same")
            #             if DEBUG:
            #                 visualize_obj(obj1, view_1_frame, "1")
            #                 visualize_obj(obj2, view_2_frame, "2")
            #                 cv2.destroyAllWindows()
            #         else:
            #             est_obj_equality.append("not-same")
            #         # ground truth object equality
            #         if obj1['track_id'] == obj2['track_id']:
            #             actual_obj_equality.append("same")
            #         else:
            #             actual_obj_equality.append("not-same")

        # find precision/recall
        prec, recall = get_prec_recall(actual=actual_obj_equality, estimated=est_obj_equality,
                                       labels=['same', 'not-same'])

        precisions.append(prec)
        recalls.append(recall)
        print("prec: {}, recall : {}".format(prec, recall))

        break

    print("Precisions : {}, recall : {}".format(precisions, recalls))

    # plot precision/recall vs threshold here
    if INCLUDE_COLOR or INCLUDE_SIFT:
        plt.plot(colorhist_thresholds, precisions, label="precision")
        plt.plot(colorhist_thresholds, recalls, label="recall")
        plt.xticks(colorhist_thresholds)
        if INCLUDE_COLOR:
            plt.xlabel("Color Histogram Threshold (Bhattacharya Distance)")
        else:
            plt.xlabel("Threshold (Normalized SIFT score)")

    elif INCLUDE_EMBEDDING:
        plt.plot(eucl_distances, precisions, label="precision")
        plt.plot(eucl_distances, recalls, label="recall")
    # plt.xticks(eucl_d)

    plt.ylabel("Precision/Recall Values")
    plt.legend()
    plt.show()
