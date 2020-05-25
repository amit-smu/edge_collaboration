"""
module to collect ground truth for object matching
"""

import pandas as pd
import cv2

if __name__ == "__main__":
    input_file = "./analysis/comparison_scores.csv"
    dataset_dir = "../dataset/"
    data = pd.read_csv(input_file)
    print(data.columns)

    for index, row in data.iterrows():
        frame_num = row['frame_num']
        view_1 = row['view_1']
        view_2 = row['view_2']
        obj_box_1 = row['obj_box_1']
        obj_box_2 = row['obj_box_2']

        frame_1_path = "{}/View_00{}/frame{}.jpg".format(dataset_dir, view_1, frame_num)
        frame_1 = cv2.imread(frame_1_path)
        box_1 = obj_box_1.split(",")
        xmin = int(box_1[2])
        ymin = int(box_1[3])
        xmax = int(box_1[4])
        ymax = int(box_1[5].strip(']'))
        obj_1 = frame_1[ymin:ymax, xmin:xmax]

        frame_2_path = "{}/View_00{}/frame{}.jpg".format(dataset_dir, view_2, frame_num)
        frame_2 = cv2.imread(frame_2_path)
        box_2 = obj_box_2.split(",")
        xmin = int(box_2[2])
        ymin = int(box_2[3])
        xmax = int(box_2[4])
        ymax = int(box_2[5].strip(']'))
        obj_2 = frame_2[ymin:ymax, xmin:xmax]

        print("{}, {}, {}, {}, {}".format(frame_num, view_1, view_2, obj_box_1, obj_box_2))
        obj_1 = cv2.resize(obj_1, dsize=None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        obj_2 = cv2.resize(obj_2, dsize=None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        cv2.imshow("obj_1", obj_1)
        cv2.imshow("obj_2", obj_2)

        cv2.waitKey(-1)
        cv2.destroyAllWindows()
