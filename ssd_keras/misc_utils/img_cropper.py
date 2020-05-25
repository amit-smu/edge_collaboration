"""
crops image from imagesets (also adjust annotations) based on given parameters
"""

import cv2
import xml.etree.cElementTree as ET
from xml.etree import ElementTree
import numpy as np
import shutil
import os
import time


def create_annotation(persons_desired, annot_path):
    annotation = ET.Element("annotation")
    ET.SubElement(annotation, "folder").text = "{}".format(DATASET)
    ET.SubElement(annotation, "filename").text = ""
    source = ET.SubElement(annotation, "source")
    ET.SubElement(source, "databse").text = "Unknown"
    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(org_img_w)
    ET.SubElement(size, "height").text = str(org_img_h)
    ET.SubElement(size, "depth").text = str(3)
    ET.SubElement(annotation, "segmented").text = "1"

    for p in persons_desired:
        object = ET.SubElement(annotation, "object")
        ET.SubElement(object, "name").text = "person"
        ET.SubElement(object, "track_id").text = "{}".format(p['personID'])
        ET.SubElement(object, "pose").text = "Frontal"
        ET.SubElement(object, "truncated").text = "1"
        ET.SubElement(object, "difficult").text = "0"
        bndbox = ET.SubElement(object, "bndbox")

        # transform bnd box coords to new resolutions
        ET.SubElement(bndbox, "xmin").text = str(p['xmin'])
        ET.SubElement(bndbox, "ymin").text = str(p['ymin'])
        ET.SubElement(bndbox, "xmax").text = str(p['xmax'])
        ET.SubElement(bndbox, "ymax").text = str(p['ymax'])

    tree = ET.ElementTree(annotation)
    tree.write(annot_path)


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

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea)
    return np.round(iou, decimals=2)


if __name__ == "__main__":
    dsize = (400, 400)  # desired size (w,h) of image
    ICOV_THRESHOLD = 0.65

    # DATASET = "WT"
    DATASET = "PETS"

    print("ICOV_THRESHOLD: {}".format(ICOV_THRESHOLD))
    print("Dsize : {}".format(dsize))
    print("Dataset : {}".format(DATASET))
    time.sleep(0.5)

    if DATASET == "WT":
        org_img_w, org_img_h = (1920, 1080)
        test_file_path = r"../../dataset/Wildtrack_dataset/ImageSets/Main/test.txt"
        imageset_dir = r"../../dataset/Wildtrack_dataset/PNGImages"
        annotation_dir = r"../../dataset/Wildtrack_dataset/Annotations"
        output_img_dir = r"../../dataset/Wildtrack_dataset/PNGImages_cropped_{}x{}".format(dsize[0], dsize[1])
        output_annot_dir = r"../../dataset/Wildtrack_dataset/Annotations_cropped_{}x{}".format(dsize[0], dsize[1])
        img_type = "png"

    if DATASET == "PETS":
        org_img_w, org_img_h = (720, 576)
        test_file_path = r"../../dataset/PETS_org/ImageSets/Main/test_12.txt"
        imageset_dir = r"../../dataset/PETS_org/JPEGImages"
        annotation_dir = r"../../dataset/PETS_org/Annotations"
        output_img_dir = r"../../dataset/PETS_org/JPEGImages_cropped_{}x{}".format(dsize[0], dsize[1])
        output_annot_dir = r"../../dataset/PETS_org/Annotations_cropped_{}x{}".format(dsize[0], dsize[1])
        img_type = "jpg"

    test_file = open(test_file_path)
    test_img_names = test_file.read().split('\n')
    print("total test images : {}\n".format(len(test_img_names)))

    # create directories
    if os.path.exists(output_img_dir):
        print("Deleting old image directory\n")
        # os.removedirs(output_img_dir)
        shutil.rmtree(output_img_dir)
    os.mkdir(output_img_dir)
    if os.path.exists(output_annot_dir):
        print("Delteing old Annotation Directory\n")
        # os.removedirs(output_annot_dir)
        shutil.rmtree(output_annot_dir)
    os.mkdir(output_annot_dir)

    for ti_name in test_img_names:
        print(ti_name)
        ATLEAST_ONE_BBOX_INCLUDED = False

        test_img = cv2.imread("{}/{}.{}".format(imageset_dir, ti_name, img_type))
        assert test_img is not None

        # crop image from center of org image
        min_x = (org_img_w // 2) - (dsize[0] // 2)
        max_x = (org_img_w // 2) + (dsize[0] // 2)
        min_y = (org_img_h // 2) - (dsize[1] // 2)
        max_y = (org_img_h // 2) + (dsize[1] // 2)

        cropped_img = test_img[min_y:max_y, min_x:max_x]
        # print("Shape of cropped_img : {}".format(cropped_img.shape))

        # cv2.imshow("cropped_img ", cropped_img)
        # cv2.waitKey(-1)

        # adjust annotation as per the crop
        test_annot = "{}/{}.xml".format(annotation_dir, ti_name)
        root = ElementTree.parse(test_annot).getroot()
        assert root is not None

        cropped_objects = []
        objects = root.findall("object")
        for obj in objects:
            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)

            # find icov score for gt box and cropped image
            icov_score = bb_icov(gt_box=[xmin, ymin, xmax, ymax], cropped_img_box=[min_x, min_y, max_x, max_y])
            if icov_score >= ICOV_THRESHOLD:
                ATLEAST_ONE_BBOX_INCLUDED = True
                cropped_objects.append({
                    'personID': obj.find('track_id').text,
                    'xmin': xmin - min_x,
                    'ymin': ymin - min_y,
                    'xmax': xmax - min_x,
                    'ymax': ymax - min_y
                })
        if ATLEAST_ONE_BBOX_INCLUDED:
            cv2.imwrite("{}/{}.{}".format(output_img_dir, ti_name, img_type), cropped_img)
            create_annotation(persons_desired=cropped_objects, annot_path="{}/{}.xml".format(output_annot_dir, ti_name))
