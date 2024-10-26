"""
find missing files in folder
"""
import os
import sys

if __name__ == "__main__":
    in_dir = r"D:\My Drive\yolov3\prior_img_model\data\Images"
    fnames = os.listdir(in_dir)

    for i in range(0, 2000, 5):
        for j in range(1, 8):
            img_name = "C{}_{:08d}.png".format(j, i)
            prior_name = "C{}_{:08d}_prior.png".format(j, i)
            label_name = "C{}_{:08d}.txt".format(j, i)

            print(img_name)
            if (img_name not in fnames) or (prior_name not in fnames) or (label_name not in fnames):
                print("\nfile missing - {}".format(img_name))
                sys.exit(-1)
