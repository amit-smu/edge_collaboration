"""
convert wildtrack xml based annotations to yolo txt based format
"""
import cv2
import os
from xml.etree import ElementTree
import numpy as np

if __name__ == "__main__":
    img_width, img_height = (1920, 1080)
    annot_dir = "dataset/Wildtrack_dataset/Annotations/"
    annot_out_dir = ""

    file_names = os.listdir(annot_dir)
    for f in file_names:
        # change annotation file
        annot_file_path = "{}/{}".format(annot_dir, f)
        output_annot_path = "{}/{}.txt".format(annot_out_dir, f[:-4])

        root = ElementTree.parse(annot_file_path).getroot()
        assert root is not None
        objects = root.findall("object")
        with open(output_annot_path, 'w') as out_file:
            for obj in objects:
                bndbox = obj.find("bndbox")
                xmin = int(bndbox.find('xmin').text)
                ymin = int(bndbox.find('ymin').text)
                xmax = int(bndbox.find('xmax').text)
                ymax = int(bndbox.find('ymax').text)

                # calculate mid point and height width of bounding box -- as per Yolo format
                width = xmax - xmin
                height = ymax - ymin
                xdiff = width / 2
                ydiff = height / 2
                mid_x = xmin + xdiff
                mid_y = ymin + ydiff
                # normalize coordinates
                mid_x = mid_x / int(img_width)
                mid_y = mid_y / int(img_height)
                width = width / int(img_width)
                height = height / int(img_height)

                # write coordinates to file
                out_file.write("{} {} {} {} {}\n".format(0, mid_x, mid_y, width, height))