"""
module to generate training/testing data for DNN which takes single croppped image as input
and outputs the bounding boxes of people inside that image. Analysis on PETS dataset
"""

import cv2
import numpy  as np
import pandas as pd
import xml.dom.minidom
import xml.etree.cElementTree as ET


# from lxml.etree import ElementTree as ET


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
    sp_overlap_dir = "{}/spatial_overlap".format(dataset_dir)
    xml_path = "{}/{}".format(sp_overlap_dir, xml_file_name)
    return get_bb_coords(xml_path=xml_path)


def write_multi_resol_img_annot(org_image, resolutions, img_output_dir, annot_output_dir, frame_num, persons_df,
                                overlp_area):
    org_height, org_width, org_channels = org_image.shape

    # write images
    for res in resolutions:
        # write image
        if (res[0] == org_width) and res[1] == org_height:
            img = org_image
        else:
            img = cv2.resize(org_image, res)
        # img_name = "frame_{}_{}x{}_{:04d}.jpg".format(collab_string, res[0], res[1], frame_num)
        img_name = "frame_{}_{:04d}.jpg".format(collab_string, frame_num)
        cv2.imwrite("{}/{}".format(img_output_dir, img_name), img)

        # write annotation files
        # annotation_name = "frame_{}_{}x{}_{:04d}.xml".format(collab_string, res[0], res[1], frame_num)
        annotation_name = "frame_{}_{:04d}.xml".format(collab_string, frame_num)

        annotation = ET.Element("annotation")
        ET.SubElement(annotation, "folder").text = "PETS"
        ET.SubElement(annotation, "filename").text = img_name
        source = ET.SubElement(annotation, "source")
        ET.SubElement(source, "databse").text = "Unknown"
        size = ET.SubElement(annotation, "size")
        ET.SubElement(size, "width").text = str(org_width)
        ET.SubElement(size, "height").text = str(org_height)
        ET.SubElement(size, "depth").text = str(org_channels)
        ET.SubElement(annotation, "segmented").text = "1"

        for index, p in persons_df.iterrows():
            object = ET.SubElement(annotation, "object")
            ET.SubElement(object, "name").text = "person"
            ET.SubElement(object, "track_id").text = "{}".format(p['track_id'])
            ET.SubElement(object, "pose").text = "Frontal"
            ET.SubElement(object, "truncated").text = "1"
            ET.SubElement(object, "difficult").text = "0"
            bndbox = ET.SubElement(object, "bndbox")

            # transform bnd box coords to new resolutions
            xmin = int((p['xmin'] / org_width) * res[0])
            xmax = int((p['xmax'] / org_width) * res[0])
            ymin = int((p['ymin'] / org_height) * res[1])
            ymax = int((p['ymax'] / org_height) * res[1])
            ET.SubElement(bndbox, "xmin").text = str(xmin)
            ET.SubElement(bndbox, "ymin").text = str(ymin)
            ET.SubElement(bndbox, "xmax").text = str(xmax)
            ET.SubElement(bndbox, "ymax").text = str(ymax)

        tree = ET.ElementTree(annotation)
        tree.write("{}/{}".format(annot_output_dir, annotation_name))


def get_icov_ratio(person_bnd_box, overlap_bnd_box):
    """
    :param person_bnd_box: person bnd box coordinates
    :param overlap_bnd_box: spatial overlap bnd box coordintates
    :return:Intersection area / person's area
    """
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(person_bnd_box[0], overlap_bnd_box[0])
    yA = max(person_bnd_box[1], overlap_bnd_box[1])
    xB = min(person_bnd_box[2], overlap_bnd_box[2])
    yB = min(person_bnd_box[3], overlap_bnd_box[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    person_area = (person_bnd_box[2] - person_bnd_box[0]) * (person_bnd_box[3] - person_bnd_box[1])

    return np.round(interArea / person_area, decimals=1)


if __name__ == "__main__":
    DEBUG = False
    dataset_dir = "../../../dataset"

    collaborating_cams = [7, 5]  # cameras that are collaborating
    collab_string_1 = "{}{}".format(collaborating_cams[0], collaborating_cams[1])
    collab_string_2 = "{}{}".format(collaborating_cams[1], collaborating_cams[0])
    # view_1_number = 7
    # view_2_number = 8
    # view_1_name = "View_00{}".format(view_1_number)
    # view_2_name = "View_00{}".format(view_2_number)

    # output directory
    # img_output_dir = "{}/model_training_data_pets/images/single_image_input".format(dataset_dir)
    img_output_dir = "{}/PETS_1/JPEGImages".format(dataset_dir)
    # annot_output_dir = "{}/model_training_data_pets/annotations/single_image_input".format(dataset_dir)
    annot_output_dir = "{}/PETS_1/Annotations".format(dataset_dir)

    image_size_pets = (720, 576)  # PETS dataset (width, height)

    # estimated spatial_overlap
    # est_sp_overlap_vw_1 = (100, 100, 300, 450)  # overlap coordinates (x1,y1,x2,y2) between vw1 and vw2 projected on vw1
    # est_sp_overlap_vw_2 = (100, 100, 300, 450)

    overlap_area_v1_v2 = get_gt_sp_overlap_coordinates("{:02d}".format(collaborating_cams[0]),
                                                       "{:02d}".format(collaborating_cams[1]))
    # overlap coordinates (x1,y1,x2,y2) between vw1 and vw2 projected on vw1
    overlap_area_v2_v1 = get_gt_sp_overlap_coordinates("{:02d}".format(collaborating_cams[1]),
                                                       "{:02d}".format(collaborating_cams[0]))
    skipped_imges = 0
    for cam_num in collaborating_cams:
        if cam_num == collaborating_cams[0]:
            overlap_area = overlap_area_v1_v2  # coordinates (x1,y1,x2,y2)
            collab_string = collab_string_1
        elif cam_num == collaborating_cams[1]:
            overlap_area = overlap_area_v2_v1
            collab_string = collab_string_2

        # compute view directory
        view_name = "View_00{}".format(cam_num)
        view_dir = "{}/{}".format(dataset_dir, view_name)

        # ground truth bounding box file. Filter ground truth
        view_grnd_truth = pd.read_csv("{}/{}.txt".format(dataset_dir, view_name), delimiter=" ")
        view_grnd_truth = view_grnd_truth[(view_grnd_truth['lost'] == 0) & (view_grnd_truth['occluded'] == 0)]

        # compute various resolutions
        org_resolution = (overlap_area[2] - overlap_area[0], overlap_area[3] - overlap_area[1])  # width,height
        resolutions = []
        for i in np.arange(0, 0.2, 0.2):
            # print(np.round(i, 1))
            r = (int(org_resolution[0] * (1 - i)), int(org_resolution[1] * (1 - i)))
            resolutions.append(r)
            # print(r)

        # read images from view dir

        for f in range(0, 795):  # frame numbers in PETS
            print("frame number : {}".format(f))
            frame_name = "frame_{:04d}".format(f)
            frame_img = cv2.imread("{}/{}.jpg".format(view_dir, frame_name))
            assert frame_img is not None

            # filter ground truth for this frame only
            persons_org = view_grnd_truth[view_grnd_truth['frame_number'] == f]

            # remove people, outside overlap area, from ground truth. Use ICov score
            persons = pd.DataFrame(columns=persons_org.columns)
            for index, p in persons_org.iterrows():
                p_bnd_box = [p['xmin'], p['ymin'], p['xmax'], p['ymax']]
                ratio = get_icov_ratio(person_bnd_box=p_bnd_box, overlap_bnd_box=overlap_area)
                if ratio >= 0.3:  # more than 30 % of the person is inside the overlap
                    persons = persons.append(persons_org.loc[index])

            if len(persons) == 0:
                skipped_imges += 1
                continue

            # reset bounding box coords as per the cropped image (they are relative values)
            persons['xmin'] = (persons['xmin'] - overlap_area[0]).clip(lower=0)

            persons['xmax'] = (persons['xmax'] - overlap_area[0]).clip(upper=overlap_area[2] - overlap_area[0])

            persons['ymin'] = (persons['ymin'] - overlap_area[1]).clip(lower=0)

            persons['ymax'] = (persons['ymax'] - overlap_area[1]).clip(upper=overlap_area[3] - overlap_area[1])

            # crop image
            crop_img = frame_img[overlap_area[1]:overlap_area[3], overlap_area[0]:overlap_area[2]]
            if DEBUG:
                cv2.imshow("crop_image", crop_img)
                cv2.waitKey(-1)

            # write multi resolution images
            write_multi_resol_img_annot(org_image=crop_img, resolutions=resolutions, img_output_dir=img_output_dir,
                                        annot_output_dir=annot_output_dir, frame_num=f, persons_df=persons,
                                        overlp_area=overlap_area)
        # break
    print("total_skipped_images : {}".format(skipped_imges))
