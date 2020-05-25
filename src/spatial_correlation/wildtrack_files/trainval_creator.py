"""
module to split and write trainval and test data from WILDTRACK dataset
"""
import numpy as np
from sklearn.model_selection import KFold


def random_frame_sampling():
    """
    randomly pick 681 frames for training (+validation) and 114 for testing the regression models
    :return:
    """
    out_dir = "../../../dataset/Wildtrack_dataset_full\Wildtrack_dataset/ImageSets"
    output_file_trainval_frames = "{}/trainval.txt".format(out_dir)
    output_file_testing_frames = "{}/test.txt".format(out_dir)
    frames_numbers = np.arange(0, 2000, dtype=np.int, step=5)

    kfold = KFold(n_splits=7, shuffle=True, random_state=10)
    for train, test in kfold.split(frames_numbers):
        # write test and train to files
        np.savetxt(fname=output_file_trainval_frames, X=frames_numbers[train], fmt="%d")
        np.savetxt(fname=output_file_testing_frames, X=frames_numbers[test], fmt="%d")
        print("training frames: {}, testing frames: {}".format(len(train), len(test)))
        break

    print("done!!")


random_frame_sampling()
