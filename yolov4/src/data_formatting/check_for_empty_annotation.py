"""
script to test if the annotation file is empty
"""
import os
import sys

in_dir = r"D:/My Drive/yolov3/single_img_model/data/Images"
names = os.listdir(in_dir)
for n in names:
    if n.__contains__("png"):
        continue
    with open("{}/{}".format(in_dir, n), "r") as f:
        print(n)
        content = f.read()
        if len(content) < 5:
            print("file empty -- {} \n".format(n))
            continue
        content = content.split("\n")
        for c in content:
            # print(c)
            c = c.split(" ")
            print("c is -- {}".format(c))
            x = float(c[1])
            y = float(c[2])
            h = float(c[3])
            w = float(c[4])
            if x == 0 or y == 0 or h == 0 or w == 0:
                print("one of the values is zero \n")
                sys.exit(-1)