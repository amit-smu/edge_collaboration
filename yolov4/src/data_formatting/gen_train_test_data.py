"""
module to generate training and test dataset files for yolo
"""
import random

if __name__ == "__main__":
    print()

    train_filename = "train.txt"
    test_filename = "test.txt"
    prefix = "../single_img_model/dataset/Wildtrack_dataset/PNGImages/"  # prefix relative to darknet directory
    cam_list = [1, 2, 3, 4, 5, 6, 7]
    file_paths = []
    with open(train_filename, 'w') as file:
        for cam in cam_list:
            for i in range(0, 1400, 5):
                file_name = "C{}_{:08d}.png".format(cam, i)
                file_paths.append("{}{}\n".format(prefix, file_name))
        # shuffle the data
        random.seed(0)
        random.shuffle(file_paths)
        for f_path in file_paths:
            file.write(f_path)

    # create test images file
    file_paths = []
    with open(test_filename, 'w') as file:
        for cam in cam_list:
            for i in range(1400, 2000, 5):
                file_name = "C{}_{:08d}.png".format(cam, i)
                file_paths.append("{}{}\n".format(prefix, file_name))
        # shuffle the data
        random.shuffle(file_paths)
        for f_path in file_paths:
            file.write(f_path)
