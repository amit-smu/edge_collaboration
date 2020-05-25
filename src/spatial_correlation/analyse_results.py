"""module to analyse results of predicted and actual person matching"""

import pandas as pd
import seaborn as sns
from pylab import boxplot, xlim
from matplotlib import pyplot as plt
import numpy as np

if __name__ == "__main__":
    input_dir = "../analysis"
    input_file = "sp_corr_grnd_truth_merged.csv"

    # read data
    data = pd.read_csv("{}/{}".format(input_dir, input_file))
    print(data.columns)
    print("total observations: {}".format(len(data)))

    # filter data
    data_same_person = data[data['same_person_gt'] == 1]
    print("total observations of same person: {}".format(len(data_same_person)))
    print("combined score statistics :\n{}".format(data_same_person['combined_score'].describe()))

    # data when persons were different
    data_diff_person = data[data['same_person_gt'] == 0]
    print("total observations of different person : {}".format(len(data_diff_person)))
    print("combined score stats : \n {}".format(data_diff_person['combined_score'].describe()))

    fig = plt.figure(figsize=(13, 8.5))
    plt.rcParams.update({'font.size': 16})
    ax = fig.add_subplot(111)

    #test
    data_same_person = data_same_person['color_score'] + np.ran

    boxplot([data_same_person['color_score'].tolist(), data_diff_person['color_score'].tolist()], widths=0.3,
            positions=[1, 2], patch_artist=True)
    boxplot([data_same_person['sift_score'].tolist(), data_diff_person['sift_score'].tolist()], widths=0.3,
            positions=[4, 5], patch_artist=True)
    boxplot([data_same_person['combined_score'].tolist(), data_diff_person['combined_score'].tolist()], widths=0.3,
            positions=[7, 8], patch_artist=True)
    boxplot([data_same_person['hu_score'].tolist(), data_diff_person['hu_score'].tolist()], widths=0.3,
            positions=[9, 10], patch_artist=True)

    ax.set_xticklabels(['color_score', 'sift_score', 'color+sift score', 'hu_score'])
    # ax.set_xticklabels(['color_score', 'sift_score', 'color+sift score'])
    ax.set_xticks([1.5, 4.5, 7.5, 9.5])
    # ax.set_xticks([1.5, 4.5, 7.5])
    xlim(0, 11)
    ax.set_ylabel("Normalized Score")
    ax.set_xlabel("Feature")

    plt.show()
