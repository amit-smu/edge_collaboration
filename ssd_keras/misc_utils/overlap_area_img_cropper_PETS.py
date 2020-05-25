"""
for given coordinates (of spatial overlap ara), it crops image from imagesets (also adjust annotations)
based on given parameters.
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


def get_overlap_area_coordinates(annot_path):
    root = ElementTree.parse(annot_path).getroot()
    assert root is not None
    objects = root.findall("object")
    for obj in objects:
        bndbox = obj.find("bndbox")
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)

    return [xmin, ymin, xmax, ymax]


def crop_img(ti_name, output_img_dir, output_annot_dir, gt_olap_area_coords):
    ATLEAST_ONE_BBOX_INCLUDED = False
    test_img = cv2.imread("{}/{}.{}".format(imageset_dir, ti_name, img_type))
    assert test_img is not None
    # crop image as per overlap coordinates
    min_x = gt_olap_area_coords[0]
    min_y = gt_olap_area_coords[1]
    max_x = gt_olap_area_coords[2]
    max_y = gt_olap_area_coords[3]
    cropped_img = test_img[min_y:max_y, min_x:max_x]
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


if __name__ == "__main__":
    ICOV_THRESHOLD = 0.65

    DATASET = "PETS"
    ref_cam = 7
    collab_cam = 8
    SP_OVERLAP_GT_DIR = r"../../dataset/spatial_overlap"

    print("ICOV_THRESHOLD: {}".format(ICOV_THRESHOLD))
    print("Dataset : {}".format(DATASET))
    time.sleep(1)

    if DATASET == "PETS":
        org_img_w, org_img_h = (720, 576)
        test_file_path = r"../../dataset/PETS_org/ImageSets/Main/test_12.txt"
        imageset_dir = r"../../dataset/PETS_org/JPEGImages"
        annotation_dir = r"../../dataset/PETS_org/Annotations"
        output_img_dir_r = r"../../dataset/PETS_org/JPEGImages_cropped_r{}_c{}_{}".format(ref_cam, collab_cam,
                                                                                          ref_cam)
        output_annot_dir_r = r"../../dataset/PETS_org/Annotations_cropped_r{}_c{}_{}".format(ref_cam,
                                                                                             collab_cam,
                                                                                             ref_cam)
        output_img_dir_c = r"../../dataset/PETS_org/JPEGImages_cropped_r{}_c{}_{}".format(ref_cam, collab_cam,
                                                                                          collab_cam)
        output_annot_dir_c = r"../../dataset/PETS_org/Annotations_cropped_r{}_c{}_{}".format(ref_cam,
                                                                                             collab_cam,
                                                                                             collab_cam)
        SP_OVERLAP_GT_DIR = "{}/{}".format(SP_OVERLAP_GT_DIR, "PETS")
        frame_prefix = "0062"
        img_prefix = "frame_"
        img_type = "jpg"

    # get coordinates of ground truth overlap area (wrt ref cam)
    gt_annot_path_r = "{}/frame_{}_{:02d}_{:02d}.xml".format(SP_OVERLAP_GT_DIR, frame_prefix, ref_cam, collab_cam)
    gt_coordinates_ref = get_overlap_area_coordinates(gt_annot_path_r)  # xmin,ymin,xmax,ymax

    gt_annot_path_c = "{}/frame_{}_{:02d}_{:02d}.xml".format(SP_OVERLAP_GT_DIR, frame_prefix, collab_cam, ref_cam)
    gt_coordinates_collab = get_overlap_area_coordinates(gt_annot_path_c)

    # test_file = open(test_file_path)
    # test_img_names = test_file.read().split('\n')
    test_img_names = range(0, 795, 1)
    print("total test images : {}\n".format(len(test_img_names)))

    # create directories
    if os.path.exists(output_img_dir_r):
        print("Deleting old image directory\n")
        shutil.rmtree(output_img_dir_r)
    os.mkdir(output_img_dir_r)
    if os.path.exists(output_annot_dir_r):
        print("Delteing old Annotation Directory\n")
        shutil.rmtree(output_annot_dir_r)
    os.mkdir(output_annot_dir_r)

    if os.path.exists(output_img_dir_c):
        print("Deleting old image directory\n")
        shutil.rmtree(output_img_dir_c)
    os.mkdir(output_img_dir_c)
    if os.path.exists(output_annot_dir_c):
        print("Delteing old Annotation Directory\n")
        shutil.rmtree(output_annot_dir_c)
    os.mkdir(output_annot_dir_c)

    for ti_name in test_img_names:
        # ti_name = "C{}_{:08d}".format(ref_cam, ti_name)
        ti_name = "frame_{}_{:04d}".format(ref_cam, ti_name)
        if not ti_name.__contains__("{}{}".format(img_prefix, ref_cam)):
            continue
        print(ti_name)
        # write images for reference camera
        crop_img(ti_name, output_img_dir_r, output_annot_dir_r, gt_coordinates_ref)

        # write images for collaborating camera
        ti_name_collab = "frame_{}_{}".format(collab_cam, ti_name[8:])
        crop_img(ti_name_collab, output_img_dir_c, output_annot_dir_c, gt_coordinates_collab)

        # delete file if not present in both folders
        ref_file_name = "{}/{}.jpg".format(output_img_dir_r, ti_name)
        collab_file_name = "{}/{}.jpg".format(output_img_dir_c, ti_name_collab)
        if os.path.exists(ref_file_name) and os.path.exists(collab_file_name):
            continue
        else:
            if os.path.exists(ref_file_name):
                print("deleting file : {}".format(ti_name))
                os.remove(ref_file_name)
            elif os.path.exists(collab_file_name):
                print("deleting file : {}".format(ti_name_collab))
                os.remove(collab_file_name)
