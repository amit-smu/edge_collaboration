"""Module to test if the data generated and annotations in ground truth are same
"""

import cv2
import os
import xml.dom.minidom
from xml.etree import ElementTree
import random


def get_bb_coords(xml_path):
    dom_tree = xml.dom.minidom.parse(xml_path)
    collection = dom_tree.documentElement
    xmin = collection.getElementsByTagName("xmin")[0].firstChild.nodeValue
    ymin = collection.getElementsByTagName("ymin")[0].firstChild.nodeValue
    xmax = collection.getElementsByTagName("xmax")[0].firstChild.nodeValue
    ymax = collection.getElementsByTagName("ymax")[0].firstChild.nodeValue
    return [int(xmin), int(ymin), int(xmax), int(ymax)]


def get_gt_sp_overlap_coordinates(view_1, view_2):
    """
    retrieves the coordinates of bounding box representing the marked spatial overlap between view 1 and view 2
    :param view_1:
    :param view_2:
    :return:
    """
    xml_file_name = "frame_0062_{}_{}.xml".format(view_1, view_2)
    sp_overlap_dir = "../../dataset/spatial_overlap"
    xml_path = "{}/{}".format(sp_overlap_dir, xml_file_name)
    return get_bb_coords(xml_path=xml_path)


if __name__ == "__main__":
    dataset_dir = "../../../dataset"
    # images_dir = "{}/{}".format(dataset_dir, "model_training_data_pets/images/single_image_input")
    # annot_dir = "{}/{}".format(dataset_dir, "model_training_data_pets/annotations/single_image_input")

    images_dir = "{}/{}".format(dataset_dir, "/PETS_1/JPEGImages/")
    annot_dir = "{}/{}".format(dataset_dir, "/PETS_1/Annotations/")

    images = os.listdir(images_dir)
    images = random.sample(population=images, k=150)
    for img_name in images:
        print(img_name)
        # img_name = "frame_87_0000.jpg"
        # if not img_name.__contains__("78"):
        #     continue
        image = cv2.imread("{}/{}".format(images_dir, img_name))
        # cv2.imshow("test", image)

        assert image is not None

        # load annotation
        annot_name = img_name.strip(".jpg") + ".xml"
        xml_path = "{}/{}".format(annot_dir, annot_name)

        root = ElementTree.parse(xml_path).getroot()
        objects = root.findall("object")
        for obj in objects:
            track_id = obj.find('track_id').text
            bndbox = obj.find('bndbox')
            xmin = int(float(bndbox[0].text))
            ymin = int(float(bndbox[1].text))
            # if obj_name == "lady":
            xmax = int(float(bndbox[2].text))
            ymax = int(float(bndbox[3].text))
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)
            cv2.putText(image, "{}".format(track_id), (xmin, ymin - 5), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 1)

        # dom_tree = xml.dom.minidom.parse(xml_path)
        # collection = dom_tree.documentElement
        # boxes = collection.getElementsByTagName("bndbox")
        # for box in boxes:
        #     xmin = int(box.childNodes[0].firstChild.nodeValue)
        #     ymin = int(box.childNodes[1].firstChild.nodeValue)
        #     xmax = int(box.childNodes[2].firstChild.nodeValue)
        #     ymax = int(box.childNodes[3].firstChild.nodeValue)
        #     print(xmin, ymin, xmax, ymax)
        #     cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.imshow("{}".format(img_name), image)
        cv2.moveWindow("{}".format(img_name), 200, 200)
        cv2.waitKey(400)
        cv2.destroyAllWindows()
        print("")
