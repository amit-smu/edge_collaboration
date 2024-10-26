"""
module to show detections from both the models side by side
"""
import os
import cv2
import numpy as np

if __name__ == "__main__":
    SINGLE_MODEL_DIR = "single"
    PRIOR_MODEL_DIR = "prior"

    single_files = os.listdir(SINGLE_MODEL_DIR)
    prior_files = os.listdir(PRIOR_MODEL_DIR)
    print(f"Total files - Singles : {len(single_files)}, Prior : {len(prior_files)}")
    # assert len(single_files) == len(prior_files)

    for file in prior_files:
        if file in single_files:

            single_img = cv2.imread(f"{SINGLE_MODEL_DIR}/{file}")
            cv2.putText(single_img, "Single Img", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
            prior_img = cv2.imread(f"{PRIOR_MODEL_DIR}/{file}")

            combined_img = np.vstack((single_img, prior_img))
            cv2.imshow("Combined image", combined_img)
            cv2.waitKey(-1)
        else:
            print(f"File {file} not present in Single detections")
