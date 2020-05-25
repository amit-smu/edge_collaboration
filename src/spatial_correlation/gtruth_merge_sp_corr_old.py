"""
merge estimated spatial correlation results with ground truth
"""
import pandas as pd

if __name__ == "__main__":
    input_dir = "../analysis"
    sp_corr_file = "sp_corr_comp_scores.csv"
    gtruth_file = "grnd_truth.csv"
    output_file = "sp_corr_grnd_truth_merged.csv"

    # read data
    sp_corr_data = pd.read_csv("{}/{}".format(input_dir, sp_corr_file))
    gtruth_data = pd.read_csv("{}/{}".format(input_dir, gtruth_file))

    # sanity checks before merging
    assert sp_corr_data['obj_box_1'].equals(gtruth_data['obj_box_1'])
    assert sp_corr_data['obj_box_2'].equals(gtruth_data['obj_box_2'])

    gtruth_data_column = gtruth_data[['same_person_gt', 'comments']]
    gtruth_data_column['same_person_gt'].replace("yes", 1, inplace=True)
    gtruth_data_column['same_person_gt'].replace("no", 0, inplace=True)
    df = pd.merge(sp_corr_data, gtruth_data_column, on=sp_corr_data.index, suffixes=("_pred", "_actual"))
    df.to_csv("{}/{}".format(input_dir, output_file), index=None)
