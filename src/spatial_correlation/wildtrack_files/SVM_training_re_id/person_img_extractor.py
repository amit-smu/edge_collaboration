"""
module to extract person images from each view to train an svm classifier for re-id like PETS
"""

import cv2

from xml.etree import ElementTree


def get_all_objects(annotation_path):
    objects = []
    doc_root = ElementTree.parse(annotation_path).getroot()
    xml_objects = doc_root.findall('object')
    for obj in xml_objects:
        obj_id = obj.find("track_id").text
        b_box = obj.find("bndbox")
        xmin = b_box[0].text
        ymin = b_box[1].text
        xmax = b_box[2].text
        ymax = b_box[3].text
        objects.append([int(obj_id), int(xmin), int(ymin), int(xmax), int(ymax)])
    return objects


if __name__ == "__main__":
    DEBUG = False
    dir_name = "../../../../dataset/Wildtrack_dataset/"
    per_person_samples = 8
    output_dir = "./person_images/"

    cameras = [1, 4, 5, 6]
    for cam in cameras:
        print("Camera : {}".format(cam))
        frame_numbers = range(0, 2000, 15)
        for f_num in frame_numbers:
            print("frame number: {}".format(f_num))
            image_path = "{}/PNGImages/C{}_{:08d}.png".format(dir_name, cam, f_num)
            annot_path = "{}/Annotations/C{}_{:08d}.xml".format(dir_name, cam, f_num)
            # find all objects in this file
            objects = get_all_objects(annot_path)
            if len(objects) == 0:
                continue
            image = cv2.imread(image_path)
            assert image is not None
            # extract persons and write to directory
            for obj in objects:
                person_id = obj[0]
                xmin, ymin, xmax, ymax = obj[1:]
                p_img = image[ymin:ymax, xmin:xmax]
                p_img_name = "{}_C{}_{:08d}.jpg".format(person_id, cam, f_num)
                cv2.imwrite("{}/{}".format(output_dir, p_img_name), p_img)
