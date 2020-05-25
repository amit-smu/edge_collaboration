import random
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    DEBUG = False
    dataset_dir = "../../../dataset/PETS_org"
    views = [5, 6, 7, 8]

    train_frames, test_frames = train_test_split(range(0, 795), test_size=0.12)
    train_frames = sorted(train_frames)
    test_frames = sorted(test_frames)

    train_file_name = "{}/ImageSets/Main/trainval_88.txt".format(dataset_dir)
    test_file_name = "{}/ImageSets/Main/test_12.txt".format(dataset_dir)

    # write train dataset
    with open(train_file_name, 'w') as train_file_output:
        print("writing trainval set")
        for v in views:
            for train_f_num in train_frames:
                frame_name = "frame_{}_{:04d}".format(v, train_f_num)
                train_file_output.write("{}\n".format(frame_name))

    # write test dataset
    with open(test_file_name, 'w') as test_file_output:
        print("writing test set..")
        for v in views:
            for test_f_num in test_frames:
                frame_name = "frame_{}_{:04d}".format(v, test_f_num)
                test_file_output.write("{}\n".format(frame_name))
