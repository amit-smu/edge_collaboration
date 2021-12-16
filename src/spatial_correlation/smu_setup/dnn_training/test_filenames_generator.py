"""
this module generates a set of training and test frame numbers used by coordinate mapper as well as DNN training and testing
"""
import os
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    IN_DIR = "../../../../rpi_hardware/raw_image_processing/data/episode_1/ground_truth/frame_wise_gt_json/cam_2"

    filenames = os.listdir(IN_DIR)
    train, test = train_test_split(filenames, test_size=0.30, random_state=50, shuffle=True)
    print(len(train), len(test))
    # dump filenames to a file
    # with open("train.txt", 'w') as out:
    #     for i in train:
    #         f_num = i[:-5].split("_")[2]
    #         out.write("frame_{}_{}.jpg\n".format(1, f_num))
    #         out.write("frame_{}_{}.jpg\n".format(2, f_num))
    #         out.write("frame_{}_{}.jpg\n".format(3, f_num))
    #
    # with open("test.txt", 'w') as out:
    #     for i in test:
    #         out.write("{}.jpg\n".format(i[:-5]))

    # dump train and test frame numbers
    with open("train_frame_numbers.txt", 'w') as out:
        for i in train:
            out.write("{}\n".format(i[:-5].split("_")[2]))

    with open("test_frame_numbers.txt", 'w') as out:
        for i in test:
            out.write("{}\n".format(i[:-5].split("_")[2]))
