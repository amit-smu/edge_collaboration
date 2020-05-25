import json
import numpy as np
import cv2
import shutil
import xml.etree.cElementTree as ET


def get_all_persons(json_path, view_num):
    # persons_json = None
    persons_desired = []
    with open(json_path) as input_file:
        persons_json = input_file.read()
    persons_json = json.loads(persons_json)

    for person in persons_json:
        views = person["views"]
        for v in views:
            if v["viewNum"] == int(view_num) - 1 and (
                    v["xmin"] >= 0 and v['xmax'] >= 0 and v['ymin'] >= 0 and v['ymax'] >= 0):
                p = {
                    "personID": person["personID"],
                    "positionID": person["positionID"],
                    "viewNum": v["viewNum"],
                    "xmin": v["xmin"],
                    "ymin": v["ymin"],
                    "xmax": v["xmax"],
                    "ymax": v["ymax"]
                }
                persons_desired.append(p)
                break

    return persons_desired


def create_annotation(persons_desired, annot_path):
    annotation = ET.Element("annotation")
    ET.SubElement(annotation, "folder").text = "WILDTRACK"
    ET.SubElement(annotation, "filename").text = ""
    source = ET.SubElement(annotation, "source")
    ET.SubElement(source, "databse").text = "Unknown"
    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(512)
    ET.SubElement(size, "height").text = str(512)
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


if __name__ == "__main__":
    # base_dir = "../../dataset/Wildtrack_dataset_full/Wildtrack_dataset/Image_subsets"
    # dst_dir = "../../dataset/Wildtrack_dataset_full/Wildtrack_dataset/PNGImages"
    # formatted_annot_dir = "../../dataset/Wildtrack_dataset_full/Wildtrack_dataset/annotations"
    # org_annot_dir = "../../dataset/Wildtrack_dataset_full/Wildtrack_dataset/annotations_positions"
    base_dir = "../../dataset/Wildtrack_dataset/Image_subsets"
    dst_dir = "../../dataset/Wildtrack_dataset/PNGImages"
    formatted_annot_dir = "../../dataset/Wildtrack_dataset/annotations"
    org_annot_dir = "../../dataset/Wildtrack_dataset/annotations_positions"

    camera_views = ["C1", "C2", "C3", "C4", "C5", "C6"]

    for view in camera_views:
        print("converting for view: {}".format(view))
        for img_num in range(0, 2000, 5):
            # copy image
            img_name = "{:08d}.png".format(img_num)
            src_img_path = "{}/{}/{}".format(base_dir, view, img_name)
            dst_img_path = "{}/{}_{}".format(dst_dir, view, img_name)
            # print(dst_img_path)
            shutil.copyfile(src=src_img_path, dst=dst_img_path)

            # ########### format annotation ################
            formatted_annot_path = "{}/{}_{}.xml".format(formatted_annot_dir, view, img_name.strip(".png"))
            org_annot_path = "{}/{}".format(org_annot_dir, img_name.strip("png") + "json")
            persons = get_all_persons(org_annot_path, view[1])
            create_annotation(persons_desired=persons, annot_path=formatted_annot_path)
