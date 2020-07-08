"""
module to find what threhold sould be used for comparing two objects for equality using the object embbeddings
"""
import numpy as np
import cv2
import pandas as pd
import random
import utils


def get_object_features(img_obj, obj_loc, obj_id):
    """
    extract features using feature extractors and create a dict
    :param img_obj:
    :param obj_id:
    :return:
    """
    features = {
        'rand_id': random.randint(10000, 99999),
        'location': obj_loc,
        'track_id': obj_id
    }
    if INCLUDE_COLOR:
        features['color'] = utils.compute_color_hist(img_obj)
    if INCLUDE_SIFT:
        features['sift'] = utils.compute_sift_features(img_obj)
    if INCLUDE_EMBEDDING:
        features['embedding'] = utils.get_obj_embedding(img_obj, temp_dir_path="../temp_files")
    return features


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


if __name__ == "__main__":
    dir_name = "../../dataset/"
    view_1_name = "View_007"
    view_2_name = "View_008"
    EUCL_THRESHOLD = 12
    image_size = (720, 576)  # PETS dataset (width, height)

    SAME_OBJECTS_ANALYSIS = False  # whether to do same obj analysis or not
    INCLUDE_COLOR = False
    INCLUDE_SIFT = False
    INCLUDE_EMBEDDING = True

    # create white images (used for marking the mapped areas)
    view_1_area = np.full((image_size[1], image_size[0]), 255, dtype=np.uint8)
    view_2_area = np.full((image_size[1], image_size[0]), 255, dtype=np.uint8)
    view_1_area_copy = view_1_area.copy()
    view_2_area_copy = view_2_area.copy()

    view_1_grnd_truth = pd.read_csv("{}/{}.txt".format(dir_name, view_1_name), delimiter=" ")
    view_2_grnd_truth = pd.read_csv("{}/{}.txt".format(dir_name, view_2_name), delimiter=" ")

    # frame_number = 1
    iou_vw1 = 0
    iou_vw2 = 0

    same_obj_eucl = []  # list containing eucl dist of same objects
    diff_obj_eucl = []  # list containing eucl dist of different objects

    for frame_number in range(0, 650, 10):  # 80% training data
        print("frame number : {}".format(frame_number))
        frame_name = "frame_{:04d}".format(frame_number)

        view_1_frame = cv2.imread("{}/{}/{}.jpg".format(dir_name, view_1_name, frame_name))
        view_2_frame = cv2.imread("{}/{}/{}.jpg".format(dir_name, view_2_name, frame_name))

        # get objects from frames (using ground truth)
        view_1_objects = view_1_grnd_truth[view_1_grnd_truth['frame_number'] == frame_number]
        view_2_objects = view_2_grnd_truth[view_2_grnd_truth['frame_number'] == frame_number]
        # apply filters on objects
        view_1_objects = view_1_objects[(view_1_objects['lost'] == 0) & (view_1_objects['occluded'] == 0)]
        view_2_objects = view_2_objects[(view_2_objects['lost'] == 0) & (view_2_objects['occluded'] == 0)]

        # compute object features for each object in frame
        view_1_obj_features = []
        view_2_obj_features = []
        view_1_obj_features = get_all_objects_features(view_1_objects, view_1_frame)
        view_2_obj_features = get_all_objects_features(view_2_objects, view_2_frame)

        # compare objects based on their track id
        for of1 in view_1_obj_features:
            for of2 in view_2_obj_features:
                euc_dist = utils.get_eucl_distance(of1['embedding'], of2['embedding'])
                if of1["track_id"] == of2["track_id"]:
                    same_obj_eucl.append(euc_dist)
                else:
                    diff_obj_eucl.append(euc_dist)

        with open("same_obj_eucls.txt", 'w') as output_file:
            for d in same_obj_eucl:
                output_file.write("{}\n".format(d))

        with open("diff_obj_eucls.txt", 'w') as output_file:
            for d in diff_obj_eucl:
                output_file.write("{}\n".format(d))
