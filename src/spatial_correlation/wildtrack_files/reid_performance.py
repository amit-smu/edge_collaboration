"""
performance evcaluation of person re-identification module for WILDTRACK dataset
"""

import cv2
import numpy as np
import utils
import pandas as pd
import random
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt


def extract_obj_features(view_num, view_img, frame_name):
    # get ground truth data (bounding box coordinates)
    gt_dir = "../../../dataset/Wildtrack_dataset_full/Wildtrack_dataset/annotations_positions"
    gt_file = "{}/{}.json".format(gt_dir, frame_name)
    gt_data = pd.read_json(gt_file)

    features = []
    for person in gt_data.iterrows():
        person = person[1]
        person_id = person["personID"]
        views = person["views"]
        for v in views:
            v_num = v['viewNum']
            if v_num + 1 == view_num:  # starts from 0
                x1, y1, x2, y2 = v["xmin"], v["ymin"], v["xmax"], v["ymax"]
                if x1 < 0 or x2 < 0 or y1 < 0 or y2 < 0:
                    # this person is not visible in this view, so break loop
                    break
                elif x1 >= image_size[0] or x2 >= image_size[0] or y1 >= image_size[1] or y2 >= image_size[1]:
                    # ignore coordinates bigger than image size
                    break
                else:
                    rand_id = random.randint(10000, 99999)
                    # extract embedding for these coordinates
                    person_img = view_img[y1:y2, x1:x2]
                    embedding = utils.get_obj_embedding(person_img, temp_dir_path="../../temp_files")
                    feature = {
                        "rand_id": rand_id,
                        "track_id": person_id,
                        "view_num": view_num,
                        "location": [(x1, y1), (x2, y2)],
                        "embedding": embedding
                    }
                    features.append(feature)
    return features


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


def match_objects(view_1_obj_features, view_2_obj_features, dist_threshold):
    est_equality = []
    actual_equality = []

    for obj_feature_1 in view_1_obj_features:
        for obj_feature_2 in view_2_obj_features:
            # estimated equality
            d = utils.get_eucl_distance(obj_feature_1["embedding"], obj_feature_2["embedding"])
            if d <= dist_threshold:
                est_equality.append("same")
            else:
                est_equality.append("not-same")
            # actual equality
            if obj_feature_1["track_id"] == obj_feature_2["track_id"]:
                actual_equality.append("same")
            else:
                actual_equality.append("not-same")
    return est_equality, actual_equality


if __name__ == "__main__":
    dir_name = "../../../dataset/Wildtrack_dataset_full\Wildtrack_dataset\Image_subsets"
    image_size = (1920, 1080)  # WILDTRACK dataset (width, height)

    ref_cam = 1
    collab_cams = [4]
    view_1_name = "C{}".format(ref_cam)
    view_2_name = "C{}".format(collab_cams[0])
    # eucl_distances = [0, 1, 2, 3, 4, 5, 10, 15, 20, 30]
    eucl_distances = [3, 5, 7, 10, 12, 15, 20]
    precisions = []
    recalls = []

    for th in eucl_distances:
        est_obj_equality = []  # "same" : when both objects are estimated to be same, not-same otherwise
        actual_obj_equality = []  # ground truth whether both objects are same or not-same
        distances = []
        for frame_number in range(0, 2000, 5):
            print("frame number : {}".format(frame_number))
            # frame_name = create_frame_name(frame_number)
            frame_name = "{:08d}".format(frame_number)

            view_1_frame = cv2.imread("{}/{}/{}.png".format(dir_name, view_1_name, frame_name))
            view_2_frame = cv2.imread("{}/{}/{}.png".format(dir_name, view_2_name, frame_name))

            view_1_obj_features = extract_obj_features(view_num=ref_cam, view_img=view_1_frame,
                                                       frame_name=frame_name)
            view_2_obj_features = extract_obj_features(view_num=collab_cams[0], view_img=view_2_frame,
                                                       frame_name=frame_name)

            est_equality, actual_equality = match_objects(view_1_obj_features, view_2_obj_features, th)
            est_obj_equality = est_obj_equality + est_equality  # list concat
            actual_obj_equality = actual_obj_equality + actual_equality

        # find precision/recall
        prec, recall = get_prec_recall(actual=actual_obj_equality, estimated=est_obj_equality,
                                       labels=['same', 'not-same'])

        precisions.append(prec)
        recalls.append(recall)
        print("threshold: {}, precision: {}, recall : {}".format(th, prec, recall))

    # plot the results
    plt.plot(eucl_distances, precisions, marker="+", label="precision")
    plt.plot(eucl_distances, recalls, marker="*", label="recall")
    plt.legend()
    plt.ylabel("Precision/Recall Values")
    plt.xlabel("Euclidean Distance Threshold")
    plt.show()
