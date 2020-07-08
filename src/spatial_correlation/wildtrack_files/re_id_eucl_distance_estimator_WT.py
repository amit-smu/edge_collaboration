"""
module to find what threhold sould be used for comparing two objects for equality using the object embbeddings
"""
import numpy as np
import cv2
import pandas as pd
import random
import utils


def extract_obj_features(view_num, view_img, frame_name):
    # print()
    # get ground truth data (bounding box coordinates)
    # gt_dir = "../../../dataset/Wildtrack_dataset_full/Wildtrack_dataset/annotations_positions"
    gt_dir = "../../../dataset/Wildtrack_dataset/annotations_positions"
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


if __name__ == "__main__":
    # dir_name = "../../../dataset/Wildtrack_dataset_full\Wildtrack_dataset\Image_subsets"
    dir_name = "../../../dataset/Wildtrack_dataset\Image_subsets"
    DEBUG = False

    image_size = (1920, 1080)  # WILDTRACK dataset (width, height)
    # SAME_OBJECTS_ANALYSIS = False  # whether to do same obj analysis or not
    # INCLUDE_COLOR = False
    # INCLUDE_SIFT = True
    INCLUDE_EMBEDDING = True

    ref_cam = 1
    collab_cams = [4, 5, 7]

    view_1_name = "C{}".format(ref_cam)
    # ref_cam_frame_est = cv2.imread("{}/{}/{}.png".format(dir_name, view_1_name, "00000005"))
    # ref_cam_frame_actual = ref_cam_frame_est.copy()

    same_obj_eucl = []  # list containing eucl dist of same objects
    diff_obj_eucl = []  # list containing eucl dist of different objects

    for c_cam in collab_cams:
        view_2_name = "C{}".format(c_cam)

        for frame_number in range(0, 1000, 10):  # 80% training data
            print("frame number : {}".format(frame_number))
            # frame_name = create_frame_name(frame_number)
            frame_name = "{:08d}".format(frame_number)

            view_1_frame = cv2.imread("{}/{}/{}.png".format(dir_name, view_1_name, frame_name))
            view_2_frame = cv2.imread("{}/{}/{}.png".format(dir_name, view_2_name, frame_name))

            # compute object features for each object in frame
            view_1_obj_features = []
            view_2_obj_features = []

            view_1_obj_features = extract_obj_features(view_num=ref_cam, view_img=view_1_frame,
                                                       frame_name=frame_name)
            view_2_obj_features = extract_obj_features(view_num=c_cam, view_img=view_2_frame,
                                                       frame_name=frame_name)

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
