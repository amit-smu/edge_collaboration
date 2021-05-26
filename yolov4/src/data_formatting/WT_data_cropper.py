"""
crop wildtrack dataset to 1056x1056 (32x33) -- yolov4 repo suggested multiple of 32
"""
import cv2
import os
from xml.etree import ElementTree
import numpy as np


def bb_icov(gt_box, cropped_img_box):
    # determine the (x, y)-coordinates of the intersection rectangle
    boxA = gt_box
    boxB = cropped_img_box
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    # boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea)
    return np.round(iou, decimals=2)


if __name__ == "__main__":
    img_dir = "dataset/Wildtrack_dataset/PNGImages/"
    annot_dir = "dataset/Wildtrack_dataset/Annotations/"
    ICOV_THRESHOLD = 0.5
    img_height, img_width = 1056, 1056

    file_names = os.listdir(img_dir)
    for f in file_names:
        f = "C1_00000795.png"
        print(f)
        img = cv2.imread("{}/{}".format(img_dir, f))

        # crop the image to 1056x1056
        img = img[12:1068, 432:1488]
        cv2.imwrite("{}/{}".format(img_dir, f), img)
        # break

        # change annotation file
        annot_file_path = "{}/{}.xml".format(annot_dir, f[:-4])
        output_annot_path = "{}/{}.txt".format(img_dir, f[:-4])
        cropped_box = [432, 12, 1488, 1068]
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

                icov = bb_icov(gt_box=[xmin, ymin, xmax, ymax], cropped_img_box=cropped_box)
                if icov >= ICOV_THRESHOLD:
                    # translate object coordinates to cropped image
                    xmin = max(0, xmin - cropped_box[0])
                    ymin = max(0, ymin - cropped_box[1])
                    xmax = min(xmax - cropped_box[0], 1056)
                    ymax = min(ymax - cropped_box[1], 1056)
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

        break