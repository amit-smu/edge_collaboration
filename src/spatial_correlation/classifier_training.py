"""
module to train a classifier for person identification problem. Instead of comparing two person images to each other, the idea
is to classify them into their respective categories and then see if they are in same category or not
"""

import pandas as pd
import numpy as np
import cv2
import pickle
import utils
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split


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


def extract_training_data(view_n_name):
    last_training_frame = 100  # number of frames kept for training
    samples_to_select = 8
    view_n_embeddings = []
    view_n_ids = []

    # load and filter the ground truth data
    view_n_grnd_truth = pd.read_csv("{}/{}.txt".format(dir_name, view_n_name), delimiter=" ")
    view_n_grnd_truth = view_n_grnd_truth[(view_n_grnd_truth['lost'] == 0) & (view_n_grnd_truth['occluded'] == 0)]
    # unique people
    person_ids = view_n_grnd_truth['track_id'].unique()

    for pid in person_ids:
        print("View : {}, person_id : {}".format(view_n_name, pid))
        p_data = view_n_grnd_truth[(view_n_grnd_truth['track_id'] == pid)]

        if len(p_data) < samples_to_select:
            random_samples = p_data
        else:
            random_samples = p_data.sample(n=samples_to_select)

        for index, row in random_samples.iterrows():
            frame_name = create_frame_name(row['frame_number'])
            view_n_frame = cv2.imread("{}/{}/{}.jpg".format(dir_name, view_n_name, frame_name))
            img = view_n_frame[row['ymin']:row['ymax'], row['xmin']:row['xmax']]
            if DEBUG:
                cv2.imshow("frame", view_n_frame)
                cv2.imshow("obj", img)

            obj_emb = utils.get_obj_embedding(img, temp_dir_path="../temp_files")

            view_n_embeddings.append(obj_emb)
            view_n_ids.append(pid)

    return view_n_embeddings, view_n_ids


if __name__ == "__main__":
    DEBUG = False
    dir_name = "../../dataset"
    view_names = ["View_005", "View_006", "View_007", "View_008", ]

    obj_embeddings = []
    obj_track_ids = []

    for v_name in view_names:
        embeddings, ids = extract_training_data(v_name)

        # add to global training data
        obj_embeddings.extend(embeddings)
        obj_track_ids.extend(ids)

    # train SVM classifier
    classifier = svm.SVC(kernel='linear')  # Linear Kernel)
    X_train, X_test, y_train, y_test = train_test_split(obj_embeddings, obj_track_ids, test_size=0.3,
                                                        random_state=109)  # 70% training and 30% test
    # train classifier
    classifier.fit(X_train, y_train)

    # Predict the response for test dataset
    y_pred = classifier.predict(X_test)

    # print("Precision:", metrics.precision_score(y_test, y_pred))
    # print("Recall:", metrics.recall_score(y_test, y_pred))

    print(metrics.classification_report(y_true=y_test, y_pred=y_pred))

    # write svm model to disk
    # filename = "svm_model_8_samples.sav"
    # pickle.dump(classifier, open(filename, 'wb'))

    # loaded_model = pickle.load(open(filename, 'rb'))
    # y_pred_loaded = loaded_model.predict(X_test)

    # print("same predictions: {}".format(y_pred is y_pred_loaded))
