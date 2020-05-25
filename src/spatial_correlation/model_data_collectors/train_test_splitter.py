"""module to randomly split dataset for training and testing"""

import os
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    dataset_dir = "../../../dataset"
    image_dir = "model_training_data_pets/images/single_image_input"
    trainval_file = "{}/model_training_data_pets/ImageSets/Main/trainval.txt".format(dataset_dir)
    train_file = "{}/model_training_data_pets/ImageSets/Main/train.txt".format(dataset_dir)
    validation_file = "{}/model_training_data_pets/ImageSets/Main/val.txt".format(dataset_dir)
    test_file = "{}/model_training_data_pets/ImageSets/Main/test.txt".format(dataset_dir)
    test_fraction = 0.25
    validation_fraction = 0.25

    all_files = os.listdir("{}/{}".format(dataset_dir, image_dir))
    total_files = len(all_files)

    print("split started..")
    # split training, testing, trainval dataset
    trainval_set, test_set = train_test_split(all_files, test_size=test_fraction, random_state=109)
    # trainval_set, test_set = train_test_split(all_files, test_size=0.25, random_state=109)
    trainval_set = sorted(trainval_set)
    test_set = sorted(test_set)

    # divide trainval into training and validation
    train_set, validation_set = train_test_split(trainval_set, test_size=validation_fraction, random_state=109)
    train_set = sorted(train_set)
    validation_set = sorted(validation_set)
    # write data to corresponding files
    with open(trainval_file, 'w') as trainval_out:
        for name in trainval_set:
            trainval_out.write("{}\n".format(name.strip(".jpg")))

    with open(train_file, 'w') as train_out:
        for name in train_set:
            train_out.write("{}\n".format(name.strip(".jpg")))

    with open(validation_file, 'w') as val_out:
        for name in validation_set:
            val_out.write("{}\n".format(name.strip(".jpg")))

    with open(test_file, 'w') as test_out:
        for name in test_set:
            test_out.write("{}\n".format(name.strip(".jpg")))
