from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, GaussianNoise
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers import MaxPooling2D, merge
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras import losses
from keras.utils import to_categorical
import keras.backend as K
from keras.preprocessing import image

import matplotlib.pyplot as plt

import numpy as np

from os import listdir
from os.path import isfile, join
from keras.models import model_from_json
from scipy.spatial import distance
import random
import cv2

from sklearn.neighbors import KNeighborsClassifier

weights_path = 'saved_model_MSMT/'
json_file = open(weights_path + 'encoder.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
encoder = model_from_json(loaded_model_json)
encoder.load_weights(weights_path + 'encoder_weights.hdf5')

json_file = open(weights_path + 'decoder.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
decoder = model_from_json(loaded_model_json)
decoder.load_weights(weights_path + 'decoder_weights.hdf5')

# class_path ='i-LIDS-VID/sequences/cam2/'
class_path = 'pets/'
classifier = KNeighborsClassifier(n_neighbors=3)
# class_path ='MSMT17_V1/train/'
classes = [f for f in listdir(class_path)]
sift = cv2.ORB_create()


def run_encoder_decoder(path):
    # self.encoder.load_weights('saved_model/encoder_weights.hdf5')
    # self.decoder.load_weights('saved_model/decoder_weights.hdf5')
    img = image.load_img(path, target_size=(128, 64))
    x = image.img_to_array(img)
    # Call the required feature calculation method..............................
    x = calculate_ColourHistogram(x)
    # x = calculate_GAN(x)
    # x = encoder.predict(x)[0]
    # gen_img = decoder.predict(x)
    # # print(gen_img.shape)
    # gen_img = 0.5 * gen_img + 0.5
    # plt.imshow(gen_img[0])
    # plt.show()
    return x


def calculate_ColourHistogram(x):
    x = cv2.cvtColor(x, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([x], [0, 1, 2], None, [5, 5, 5], [0, 256, 0, 256, 0, 256])
    return hist.flatten();


def calculate_GAN(x):
    x = np.expand_dims(x, axis=0)
    x = encoder.predict(x)[0]
    return x


def calc_best_match(query, database, pointer, database_len):
    minimum = 0;
    score_min = 100000000.0
    # score_min = -100000000.0
    min_ind = 0;
    i = pointer;
    for ind in database:

        # score = sum(sum((ind-query)**2))
        # score = distance.correlation(ind,query)
        # score = distance.russellrao(ind,query)
        score = distance.cdist(ind, query)
        # print(score)
        # score = distance.mahalanobis(query, ind,None)
        # print(score)
        if (score < score_min):
            score_min = score;
            min_ind = i;
        i = i + 1;
    # print(query)
    # 0.2 #0.4
    length = database_len[min_ind - pointer] * 1.0
    # if (score_min < 0.2):
    #     database[min_ind-pointer] = (length*database[min_ind-pointer])/(length+1.0) + query/(length+1.0)  #0.9 
    #     database_len[min_ind-pointer] = length + 1.0
    # if (score_min > 0.4):
    #     min_ind = None
    return score_min, min_ind;


# database_path = 'i-LIDS-VID/sequences/cam2/person'
database_path = 'pets/person'
database = []
database_len = []
X_train = []
Y_train = []
min_ind = 1;
max_ind = 13;

# for i in range(min_ind,max_ind):
# classes= classes[280:300]
for class_ in classes:
    # file_path = database_path+'{num:03d}'.format(num=i)+'/'
    print(class_)
    file_path = class_path + class_ + '/'
    files = [join(file_path, f) for f in listdir(file_path) if isfile(join(file_path, f))]
    # database.append(0.25*(calc_features(files[0],pca)+calc_features(files[1],pca)+calc_features(files[2],pca)+calc_features(files[3],pca)));
    # database.append(calc_features(files[0],pca))
    # database.append(1.0*run_encoder_decoder(files[0]) + 0.25*run_encoder_decoder(files[1])+ 0.25*run_encoder_decoder(files[2])+ 0.25*run_encoder_decoder(files[3]))

    dat_val = 0
    for k in range(100):
        # dat_val += (1/40.0)*run_encoder_decoder(files[k])
        # print(run_encoder_decoder(files[k])[0].shape)
        seed = random.randint(0, len(files) - 1)
        X_train.append(run_encoder_decoder(files[seed]))
        Y_train.append(class_)
    database_len.append(40.0)
# query_main = 'i-LIDS-VID/sequences/cam2/person'
classifier.fit(np.array(X_train), np.array(Y_train))
query_main = 'pets/person'
score = 0;
false_positive = 0;
length = 0;

# for i in range(min_ind,max_ind):
for class_ in classes:
    # query_path = query_main +'{num:03d}'.format(num=i)+'/'
    query_path = class_path + class_ + '/'
    files = [join(query_path, f) for f in listdir(query_path) if isfile(join(query_path, f))]

    # score = 0
    # false_positive = 0

    for file in files:
        query_features = run_encoder_decoder(file);
        y_pred = classifier.predict(np.array(query_features).reshape(1, -1))
        print(y_pred)
        # val = calc_best_match(query_features,database,0,database_len)
        if class_ == y_pred[0]:
            score += 1
            print(score, 'Detected: ', y_pred[0], 'Ground Truth: ', class_)
        else:
            print("False:")
            false_positive += 1
        # if (y_pred[0]!=None):
        #     if  classes[val[1]] == class_:
        #         score +=1
        #         print(score,' distance:', val[0], 'Detected: ', classes[val[1]], 'Ground Truth: ', class_)
        #     else:
        #         print("False: "' distance:', val[0])
        #         false_positive +=1
    length += len(files)

accuracy = (score / length * 100.0)
false_positive = (false_positive / length * 100.0)
print(accuracy)
print(false_positive)
# print(calc_best_match(query_features,database))


# run_encoder_decoder('bbox_test/0002/0002C1T0001F012.jpg')
