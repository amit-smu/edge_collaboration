"""
use the extracted person iamges and train a svm classifier on them for RE-ID

"""

import cv2
import pickle
# import utils
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
import os
import random
from src.server_files import dnn_client as dc


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


def select_images():
    obj_embeddings = []
    obj_track_ids = []

    filenames = os.listdir(dir_name)
    # find unique number of people
    p_ids = []
    for f_name in filenames:
        f_name = f_name.split("_")[0]
        p_ids.append(f_name)
    p_ids = list(set(p_ids))
    # select images for each person id

    for pid in p_ids:
        pid = int(pid)
        if pid > 50:
            continue
        print("person id : {}".format(pid))
        for cam in cameras:
            print("camera : {}".format(cam))
            files = [f for f in filenames if f.startswith("{}_C{}".format(pid, cam))]
            if len(files) > imags_per_camera:
                files = random.sample(files, k=imags_per_camera)
            # get obj embedding sfor these iamges
            for f in files:
                img = cv2.imread("{}/{}".format(dir_name, f))
                obj_emb = get_obj_embedding(img, temp_dir_path="./temp_files")
                obj_embeddings.append(obj_emb)
                obj_track_ids.append(int(pid))

    return obj_embeddings, obj_track_ids


if __name__ == "__main__":
    DEBUG = False
    dir_name = "person_images/"
    cameras = [1, 4]
    imags_per_camera = 15

    obj_embeddings, obj_track_ids = select_images()
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
    filename = "svm_model_8_samples_WT.sav"
    pickle.dump(classifier, open(filename, 'wb'))

    loaded_model = pickle.load(open(filename, 'rb'))
    y_pred_loaded = loaded_model.predict(X_test)

    print("same predictions: {}".format(y_pred is y_pred_loaded))
