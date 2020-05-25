"""
convert PETS dataset (original) to a format that SSD can use (xml)
"""

import cv2
import numpy  as np
import pandas as pd
import xml.dom.minidom
import xml.etree.cElementTree as ET
import shutil
import os


def create_annot_file(annot_path, df):
    annotation = ET.Element("annotation")
    ET.SubElement(annotation, "folder").text = "PETS"
    ET.SubElement(annotation, "filename").text = ""
    source = ET.SubElement(annotation, "source")
    ET.SubElement(source, "databse").text = "Unknown"
    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(720)
    ET.SubElement(size, "height").text = str(576)
    ET.SubElement(size, "depth").text = str(3)
    ET.SubElement(annotation, "segmented").text = "1"

    for index, p in df.iterrows():
        object = ET.SubElement(annotation, "object")
        ET.SubElement(object, "name").text = "person"
        ET.SubElement(object, "track_id").text = "{}".format(p['track_id'])
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


if __name__ == "__main__":
    DEBUG = False
    dataset_dir = "../../../dataset/PETS_org"

    views = [5, 6, 7, 8]
    dst_dir = "{}/JPEGImages".format(dataset_dir)

    for v in views:
        view_name = "View_00{}".format(v)

        # list all files
        all_files = os.listdir("{}/{}".format(dataset_dir, view_name))

        # ground truth bounding box file. Filter ground truth
        view_grnd_truth = pd.read_csv("{}/{}.txt".format(dataset_dir, view_name), delimiter=" ")
        view_grnd_truth = view_grnd_truth[(view_grnd_truth['lost'] == 0) & (view_grnd_truth['occluded'] == 0)]

        for f in range(0, 795):  # frame numbers in PETS
            print("frame number : {}".format(f))
            frame_name = "frame_{:04d}".format(f)
            # frame_img = cv2.imread("{}/{}.jpg".format(view_dir, frame_name))
            # assert frame_img is not None
            img_path = "{}/{}/{}.jpg".format(dataset_dir, view_name, frame_name)

            # copy & rename image file
            shutil.copy(src=img_path, dst=dst_dir)
            dst_file_old_name = "{}/{}.jpg".format(dst_dir, frame_name)
            f_new = "frame_{}_{:04d}".format(v, f)
            dst_file_new_name = "{}/{}.jpg".format(dst_dir, f_new)
            os.rename(src=dst_file_old_name, dst=dst_file_new_name)

            # create annotation file
            # filter ground truth for this frame only
            persons_org = view_grnd_truth[view_grnd_truth['frame_number'] == f]
            annot_name = "{}.xml".format(f_new)
            annot_path = "{}/{}/{}".format(dataset_dir, "Annotations", annot_name)
            create_annot_file(annot_path=annot_path, df=persons_org)

            # break
        # break

        # # for f in all_files:
        # #     # copy file
        # #     print(f)
        # #     src_path = "{}/{}/{}".format(dataset_dir, view_name, f)
        # #
        # #     shutil.copy(src=src_path, dst=dst_dir)
        # #     dst_file_old_name = "{}/{}".format(dst_dir, f)
        # #     f_new = "frame_{}_{}".format(v, f[6:])
        # #     dst_file_new_name = "{}/{}".format(dst_dir, f_new)
        # #     os.rename(src=dst_file_old_name, dst=dst_file_new_name)
        # #     break
        # break
