"""
module to strip 4th channel from WT images and save it as single channel image
"""

import os
import cv2

if __name__ == "__main__":
    in_dir = "./data/Images/"
    out_dir = "./data/Images_1"
    fnames = os.listdir(in_dir)
    for name in fnames:
        if name.__contains__("txt"):
            continue
        print(name)
        image = cv2.imread("{}/{}".format(in_dir, name), cv2.IMREAD_UNCHANGED)
        assert image is not None

        image_3_channels = image[:, :, 0:3]
        prior = image[:, :, 3]

        cv2.imwrite("{}/{}".format(out_dir, name), image_3_channels)

        prior_name = name.split(".png")[0] + "_prior" + ".png"

        cv2.imwrite("{}/{}".format(out_dir, prior_name), prior)
        # break
