"""
find spatial overlap between a set of cameras. Take one reference camera and for each camera pair with this reference
camera, find the spatial overlap using techniques used in "spatial_correlation.py" file
"""

import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import random
import utils
import xml.dom.minidom


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
    xml_file_name = "frame_000000_{:02d}_{:02d}.xml".format(view_1, view_2)
    sp_overlap_dir = "../../../dataset/spatial_overlap/WILDTRACK/"
    xml_path = "{}/{}".format(sp_overlap_dir, xml_file_name)
    return get_bb_coords(xml_path=xml_path)


def get_object_features(img_obj, obj_loc, obj_id):
    """
    extract features using feature extractors and create a dict
    :param img_obj:
    :param obj_id:
    :return:
    """
    features = {
        'rand_id': random.randint(10000, 99999),
        'location': obj_loc,
        'track_id': obj_id
    }
    # if INCLUDE_COLOR:
    #     features['color'] = utils.compute_color_hist(img_obj)
    # if INCLUDE_SIFT:
    #     features['sift'] = utils.compute_sift_features(img_obj)
    if INCLUDE_EMBEDDING:
        features['embedding'] = utils.get_obj_embedding(img_obj, temp_dir_path="../temp_files")
    return features


def get_all_objects_features(view_n_objects, view_n_frame):
    """

    :param view_n_objects:
    :param view_n_frame:
    :return:
    """
    view_n_obj_features = []
    for _, obj in view_n_objects.iterrows():
        img = view_n_frame[obj['ymin']:obj['ymax'], obj['xmin']:obj['xmax']]
        obj_location = [(obj['xmin'], obj['ymin']), (obj['xmax'], obj['ymax'])]
        obj_track_id = obj['track_id']
        view_n_obj_features.append(get_object_features(img, obj_location, obj_track_id))

    return view_n_obj_features


def match_objects(objects_1, objects_2):
    """
    matches objects in one set to teh objects in other set, finds best possible match using the compare_objects method
    :param objects_1:
    :param objects_2:
    :return:
    """
    object_mappings = []

    for obj1 in objects_1:
        # best_score = 1000
        # matching_obj = -1
        for obj2 in objects_2:
            # color_score, sift_score, eucl_distance = get_scores(obj1, obj2)
            # color_score, sift_score, eucl_distance = get_scores(obj1, obj2)

            # total_score = sift_score + (1 - color_score)
            # total_score = eucl_distance
            # if total_score < best_score:
            #     matching_obj = obj2
            #     best_score = total_score
            euc_dist = utils.get_eucl_distance(obj1['embedding'], obj2['embedding'])
            if euc_dist <= 10:
                obj_same = True
            else:
                obj_same = False
            # obj_same = utils.same_objects(obj1['embedding'], obj2['embedding'])
            if obj_same:
                # add it to dataframe
                object_mappings.append(
                    [obj1['rand_id'], obj2['rand_id'], 1.0, obj1['location'], obj2['location']])
                break

    dataframe = None
    if len(object_mappings) > 0:
        dataframe = pd.DataFrame(object_mappings,
                                 columns=['obj_id_1', 'obj_id_2', 'matching_score', 'obj_id_1_loc',
                                          'obj_id_2_loc'])
    else:
        return None
    # select the pairs with maximum matching score (in case of 1 to many mappings)
    # final_obj_mappings = []
    final_obj_mappings_df = pd.DataFrame(columns=['obj_id_1', 'obj_id_2', 'matching_score', 'obj_id_1_loc',
                                                  'obj_id_2_loc'])
    df_groups = dataframe.groupby(by=dataframe['obj_id_2'])
    for grp_name, group in df_groups:
        max_score_index = group['matching_score'].idxmax()
        # obj_map = dataframe.iloc[max_score_index].values
        final_obj_mappings_df = final_obj_mappings_df.append(dataframe.iloc[max_score_index])
        # final_obj_mappings.append([int(obj_map[0]), int(obj_map[1]), obj_map[2]])
    # return final_obj_mappings
    return final_obj_mappings_df


def mark_matching_obj_area(obj_mappings, view_1_area, view_2_area, view_1_obj_features, view_2_obj_features):
    """
    for a given mapping of objects in view 1 to objects in view 2, this function will create regions of different
    intensities based on how many objects matched within a given area
    :param obj_mappings:
    :return:
    """
    delta = 10

    # view_1_area_copy = view_1_area.copy()
    # view_2_area_copy = view_2_area.copy()

    # mark matched objects areas (decrease pixel values by delta)
    for index, row in obj_mappings.iterrows():
        # objects in view 1
        obj1_loc = row['obj_id_1_loc']
        x1, y1 = obj1_loc[0]
        x2, y2 = obj1_loc[1]

        for r in range(y1, y2):
            for c in range(x1, x2):
                pixel_value = np.int(view_1_area[r, c])
                if pixel_value - delta < 0:
                    view_1_area[r, c] = 0
                    # view_1_area_copy[r, c] = 0
                else:
                    view_1_area[r, c] = view_1_area[r, c] - delta
                    # view_1_area_copy[r, c] = view_1_area_copy[r, c] - delta

        # objects in view 2
        obj2_loc = row['obj_id_2_loc']
        x1, y1 = obj2_loc[0]
        x2, y2 = obj2_loc[1]
        for r in range(y1, y2):
            for c in range(x1, x2):
                pixel_value = np.int(view_2_area[r, c])
                if pixel_value - delta < 0:
                    view_2_area[r, c] = 0
                    # view_2_area_copy[r, c] = 0
                else:
                    view_2_area[r, c] = view_2_area[r, c] - delta
                    # view_2_area_copy[r, c] = view_2_area_copy[r, c] - delta

        if DEBUG:
            cv2.imshow("view_1_area", view_1_area)
            cv2.imshow("view_2_area", view_2_area)
            # cv2.imshow("view_1_area_copy", view_1_area_copy)
            # cv2.imshow("view_2_area_copy", view_2_area_copy)

    # mark unmatched areas (increase pixel values by delta) (penalty for not matching)
    for obj1 in view_1_obj_features:
        d = obj_mappings[obj_mappings['obj_id_1'] == obj1['rand_id']]
        if len(d) == 0:
            # object not matched, so mark it
            obj1_loc = obj1['location']
            x1, y1 = obj1_loc[0]
            x2, y2 = obj1_loc[1]
            for r in range(y1, y2):
                for c in range(x1, x2):
                    pixel_value = np.int(view_1_area[r, c])
                    if pixel_value + delta > 255:
                        view_1_area[r, c] = 255
                    else:
                        view_1_area[r, c] = view_1_area[r, c] + delta

    for obj2 in view_2_obj_features:
        d = obj_mappings[obj_mappings['obj_id_2'] == obj2['rand_id']]
        if len(d) == 0:
            # object not matched, so mark it
            obj2_loc = obj2['location']
            x1, y1 = obj2_loc[0]
            x2, y2 = obj2_loc[1]
            for r in range(y1, y2):
                for c in range(x1, x2):
                    pixel_value = np.int(view_2_area[r, c])
                    if pixel_value + delta > 255:
                        view_2_area[r, c] = 255
                    else:
                        view_2_area[r, c] = view_2_area[r, c] + delta

    if DEBUG:
        cv2.imshow("view_1_area", view_1_area)
        cv2.imshow("view_2_area", view_2_area)


def get_fitting_rect_coordinates(vw_objs_features, dataframe, column):
    """
    returns the coordiantes of a rectangle around the points in given column of the dataframe
    :param dataframe:
    :param column:
    :return:
    """
    # obj_id_list = dataframe[column].values
    # for id in obj_id_list:
    #     id = int(id)
    #     for obj in vw_objs_features:
    #         if obj['rand_id'] == id:
    #             obj_coordinates.append(obj['location'])
    #             break
    # find fitting rectrange coordinates
    col_name = "{}_loc".format(column)
    obj_coordinates = dataframe[col_name].values

    min_x_list = [x[0][0] for x in obj_coordinates]
    min_x = min(min_x_list)
    max_x_list = [x[1][0] for x in obj_coordinates]
    max_x = max(max_x_list)
    min_y_list = [x[0][1] for x in obj_coordinates]
    min_y = min(min_y_list)
    max_y_list = [x[1][1] for x in obj_coordinates]
    max_y = max(max_y_list)

    return [min_x, min_y, max_x, max_y]


def update_est_sp_overlap_area(current_box_coords, prev_box_coords):
    """
    update the spatial overlap area using the area coordinates estimated from previous frames and current frame
    :param prev_box_coords:
    :param current_box_coords:
    :return:
    """
    # global est_box_coords_vw1_global
    # prev_box_coords = est_box_coords_vw1_global
    # find left most points
    left_min_x = min(prev_box_coords[0], prev_box_coords[2], current_box_coords[0], current_box_coords[2])
    left_min_y = min(prev_box_coords[1], prev_box_coords[3], current_box_coords[1], current_box_coords[3])
    right_max_x = max(prev_box_coords[0], prev_box_coords[2], current_box_coords[0], current_box_coords[2])
    right_max_y = max(prev_box_coords[1], prev_box_coords[3], current_box_coords[1], current_box_coords[3])
    return [left_min_x, left_min_y, right_max_x, right_max_y]


def extract_obj_features(view_num, view_img, frame_name):
    # print()
    # get ground truth data (bounding box coordinates)
    # gt_dir = "../../../dataset/Wildtrack_dataset_full/Wildtrack_dataset/annotations_positions"
    gt_dir = "../../../dataset/Wildtrack_dataset/annotations_positions"
    gt_file = "{}/{}.json".format(gt_dir, frame_name)
    gt_data = pd.read_json(gt_file)

    features = []
    for person in gt_data.iterrows():
        person = person[1]
        person_id = person["personID"]
        views = person["views"]
        for v in views:
            v_num = v['viewNum']
            if v_num + 1 == view_num:  # starts from 0
                x1, y1, x2, y2 = v["xmin"], v["ymin"], v["xmax"], v["ymax"]
                if x1 < 0 or x2 < 0 or y1 < 0 or y2 < 0:
                    # this person is not visible in this view, so break loop
                    break
                elif x1 >= image_size[0] or x2 >= image_size[0] or y1 >= image_size[1] or y2 >= image_size[1]:
                    # ignore coordinates bigger than image size
                    break
                else:
                    rand_id = random.randint(10000, 99999)
                    # extract embedding for these coordinates
                    person_img = view_img[y1:y2, x1:x2]
                    embedding = utils.get_obj_embedding(person_img, temp_dir_path="../../temp_files")
                    feature = {
                        "rand_id": rand_id,
                        "track_id": person_id,
                        "view_num": view_num,
                        "location": [(x1, y1), (x2, y2)],
                        "embedding": embedding
                    }
                    features.append(feature)

    return features


if __name__ == "__main__":
    # dir_name = "../../../dataset/Wildtrack_dataset_full\Wildtrack_dataset\Image_subsets"
    dir_name = "../../../dataset/Wildtrack_dataset\Image_subsets"
    # view_1_name = "View_007"
    # view_2_name = "View_008"
    # EUCL_THRESHOLD = 12
    DEBUG = False

    image_size = (1920, 1080)  # WILDTRACK dataset (width, height)
    # SAME_OBJECTS_ANALYSIS = False  # whether to do same obj analysis or not
    # INCLUDE_COLOR = False
    # INCLUDE_SIFT = True
    INCLUDE_EMBEDDING = True

    ref_cam = 1
    collab_cams = [4]
    colors = [(255, 255, 0), (0, 255, 0), (0, 0, 255)]
    color_index = 0

    view_1_name = "C{}".format(ref_cam)
    ref_cam_frame_est = cv2.imread("{}/{}/{}.png".format(dir_name, view_1_name, "00000005"))
    ref_cam_frame_actual = ref_cam_frame_est.copy()

    for c_cam in collab_cams:
        view_2_name = "C{}".format(c_cam)

        # create white images (used for marking the mapped areas)

        view_1_area = np.full((image_size[1], image_size[0]), 255, dtype=np.uint8)
        view_2_area = np.full((image_size[1], image_size[0]), 255, dtype=np.uint8)
        view_1_area_copy = view_1_area.copy()
        view_2_area_copy = view_2_area.copy()

        # ground truth spatial overlap area coordinates
        # gt_box_coords_vw_1 = [100, 200, 300, 350]  # [x1,y1,x2,y2]
        gt_box_coords_vw_1 = get_gt_sp_overlap_coordinates(ref_cam, c_cam)
        gt_box_coords_vw_2 = get_gt_sp_overlap_coordinates(c_cam, ref_cam)

        # estimated spatial overlap area coordinates
        est_box_coords_vw1_global = []
        est_box_coords_vw2_global = []

        # iou values for most recent estimated spatial overlap frames
        iou_values_1 = []
        iou_values_2 = []

        # view_1_grnd_truth = pd.read_csv("{}/{}.txt".format(dir_name, view_1_name), delimiter=" ")
        # view_2_grnd_truth = pd.read_csv("{}/{}.txt".format(dir_name, view_2_name), delimiter=" ")

        # frame_number = 1
        for frame_number in range(0, 2000, 5):
            print("frame number : {}".format(frame_number))
            # frame_name = create_frame_name(frame_number)
            frame_name = "{:08d}".format(frame_number)

            view_1_frame = cv2.imread("{}/{}/{}.png".format(dir_name, view_1_name, frame_name))
            view_2_frame = cv2.imread("{}/{}/{}.png".format(dir_name, view_2_name, frame_name))

            # ############################# Replace with object Detector #################################
            # get objects from frames (using ground truth)
            # view_1_objects = view_1_grnd_truth[view_1_grnd_truth['frame_number'] == frame_number]
            # view_2_objects = view_2_grnd_truth[view_2_grnd_truth['frame_number'] == frame_number]
            # # apply filters on objects
            # view_1_objects = view_1_objects[(view_1_objects['lost'] == 0) & (view_1_objects['occluded'] == 0)]
            # view_2_objects = view_2_objects[(view_2_objects['lost'] == 0) & (view_2_objects['occluded'] == 0)]

            # compute object features for each object in frame
            view_1_obj_features = []
            view_2_obj_features = []

            view_1_obj_features = extract_obj_features(view_num=ref_cam, view_img=view_1_frame,
                                                       frame_name=frame_name)
            view_2_obj_features = extract_obj_features(view_num=c_cam, view_img=view_2_frame,
                                                       frame_name=frame_name)
            # print("view_1_features: {}, view_2_obj_features: {}".format(len(view_1_obj_features),
            #                                                             len(view_2_obj_features)))
            # for _, obj in view_1_objects.iterrows():
            #     img = view_1_frame[obj['ymin']:obj['ymax'], obj['xmin']:obj['xmax']]
            #     obj_location = [(obj['xmin'], obj['ymin']), (obj['xmax'], obj['ymax'])]
            #     view_1_obj_features.append(get_object_features(img, obj_location))
            #
            # for _, obj in view_2_objects.iterrows():
            #     img = view_2_frame[obj['ymin']:obj['ymax'], obj['xmin']:obj['xmax']]
            #     obj_location = [(obj['xmin'], obj['ymin']), (obj['xmax'], obj['ymax'])]
            #     view_2_obj_features.append(get_object_features(img, obj_location))

            # view_1_obj_features = get_all_objects_features(view_1_objects, view_1_frame)
            # view_2_obj_features = get_all_objects_features(view_2_objects, view_2_frame)

            # compare objects from view 1 with objects from view 2
            vw_1_matching_obj_coord = []
            vw_2_matching_obj_coord = []

            obj_mappings = match_objects(view_1_obj_features, view_2_obj_features)

            if obj_mappings is None:
                continue
            # mark overlapping areas
            mark_matching_obj_area(obj_mappings=obj_mappings, view_1_area=view_1_area, view_2_area=view_2_area,
                                   view_1_obj_features=view_1_obj_features, view_2_obj_features=view_2_obj_features)
            # save marked area for this loop
            cv2.imwrite("intermediate_frames/marked_area_cam_r1_c{}_f_{}.jpg".format(c_cam, frame_number), view_1_area)

            # visualize matched objects
            if DEBUG:
                for index, row in obj_mappings.iterrows():
                    rand_num = random.randint(10000, 99999)
                    score = np.round(row['matching_score'], 2)
                    obj1_loc = row['obj_id_1_loc']
                    cv2.rectangle(view_1_frame, (obj1_loc[0]), (obj1_loc[1]), (255, 255, 0), 4)
                    cv2.putText(view_1_frame, "{}_{}".format(rand_num, score), (obj1_loc[0][0], obj1_loc[0][1] - 10),
                                cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 0), 1)

                    obj2_loc = row['obj_id_2_loc']
                    cv2.rectangle(view_2_frame, (obj2_loc[0]), (obj2_loc[1]), (255, 255, 0), 4)
                    cv2.putText(view_2_frame, "{}_{}".format(rand_num, score), (obj2_loc[0][0], obj2_loc[0][1] - 10),
                                cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 0), 1)

            # common area between the 2 views
            # min_x_1, max_x_1, min_y_1, max_y_1 = get_min_max_coords(vw_1_matching_obj_coord)
            est_box_coords_vw1 = get_fitting_rect_coordinates(view_1_obj_features, dataframe=obj_mappings,
                                                              column="obj_id_1")
            # min_x_2, max_x_2, min_y_2, max_y_2 = get_min_max_coords(vw_2_matching_obj_coord)
            est_box_coords_vw2 = get_fitting_rect_coordinates(view_2_obj_features, dataframe=obj_mappings,
                                                              column="obj_id_2")

            # # draw polygons
            # poly_pts_v_1 = np.array([[min_x_1, min_y_1], [max_x_1, max_y_1]], np.int32)
            # poly_pts_v_1 = np.array([[min_x_1, min_y_1], [max_x_1, max_y_1]], np.int32)
            # add newly found spatial area coordinates to global list (created using previous frames)
            if len(est_box_coords_vw1_global) > 0:
                est_box_coords_vw1_global = update_est_sp_overlap_area(current_box_coords=est_box_coords_vw1,
                                                                       prev_box_coords=est_box_coords_vw1_global)
            else:
                est_box_coords_vw1_global = est_box_coords_vw1

            if len(est_box_coords_vw2_global) > 0:
                est_box_coords_vw2_global = update_est_sp_overlap_area(current_box_coords=est_box_coords_vw2,
                                                                       prev_box_coords=est_box_coords_vw2_global)
            else:
                est_box_coords_vw2_global = est_box_coords_vw2

            # find IoU of ground truth spatial overlap coords with estimated spatial overlap coords
            iou_vw1 = utils.bb_iou(gt_box_coords_vw_1, est_box_coords_vw1_global)
            iou_values_1.append(iou_vw1)
            iou_vw2 = utils.bb_iou(gt_box_coords_vw_2, est_box_coords_vw2_global)
            iou_values_2.append(iou_vw2)

            if DEBUG:
                # draw_rect(gt_box_coords_vw_1, view_1_frame, (0, 255, 0))
                cv2.rectangle(view_1_frame, (gt_box_coords_vw_1[0], gt_box_coords_vw_1[1]),
                              (gt_box_coords_vw_1[2], gt_box_coords_vw_1[3]), (0, 255, 0), 2)
                cv2.putText(view_1_frame, "ground_truth", (gt_box_coords_vw_1[0], gt_box_coords_vw_1[1] - 10),
                            cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, (0, 255, 0), 1)

                # draw_rect(est_box_coords_vw1_global, view_1_frame, (0, 0, 255))
                cv2.rectangle(view_1_frame, (est_box_coords_vw1_global[0], est_box_coords_vw1_global[1]),
                              (est_box_coords_vw1_global[2], est_box_coords_vw1_global[3]), (0, 0, 255), 2)

                cv2.putText(view_1_frame, "estimated overlap",
                            (est_box_coords_vw1_global[0], est_box_coords_vw1_global[1] - 10),
                            cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, (0, 0, 255), 1)
                cv2.imshow("Camera_1", view_1_frame)

                # draw_rect(gt_box_coords_vw_2, view_2_frame, (0, 255, 0))
                cv2.rectangle(view_2_frame, (gt_box_coords_vw_2[0], gt_box_coords_vw_2[1]),
                              (gt_box_coords_vw_2[2], gt_box_coords_vw_2[3]), (0, 255, 0), 2)

                cv2.putText(view_2_frame, "ground_truth", (gt_box_coords_vw_2[0], gt_box_coords_vw_2[1] - 10),
                            cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, (0, 255, 0), 1)
                # draw_rect(est_box_coords_vw2_global, view_2_frame, (0, 0, 255))
                cv2.rectangle(view_2_frame, (est_box_coords_vw2_global[0], est_box_coords_vw2_global[1]),
                              (est_box_coords_vw2_global[2], est_box_coords_vw2_global[3]), (0, 0, 255), 2)

                cv2.putText(view_2_frame, "estimated overlap",
                            (est_box_coords_vw2_global[0], est_box_coords_vw2_global[1] - 10),
                            cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, (0, 0, 255), 1)
                cv2.imshow("Camera_2", view_2_frame)

                cv2.waitKey(-1)

        # ############## ############### draw final estimated area and actual overlap area on a frame from reference camera ####################
        cv2.rectangle(ref_cam_frame_est, (est_box_coords_vw1_global[0], est_box_coords_vw1_global[1]),
                      (est_box_coords_vw1_global[2], est_box_coords_vw1_global[3]), colors[color_index], 2)
        cv2.putText(ref_cam_frame_est, "Estimated Overlap",
                    (est_box_coords_vw1_global[0] + 20, est_box_coords_vw1_global[1] - 5),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    1.2, colors[color_index], 2)
        color_index += 1
        cv2.rectangle(ref_cam_frame_est, (gt_box_coords_vw_1[0], gt_box_coords_vw_1[1]),
                      (gt_box_coords_vw_1[2], gt_box_coords_vw_1[3]), colors[color_index], 2)
        cv2.putText(ref_cam_frame_est, "Actual Overlap",
                    (gt_box_coords_vw_1[0] + 20, gt_box_coords_vw_1[1] + 15), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    1.2, colors[color_index], 2)
        # color_index += 1

        cv2.imshow("ref_cam_frame_est", ref_cam_frame_est)
        # cv2.imshow("ref_cam_frame_actual", ref_cam_frame_actual)

        # draw actual overlap area
        # cv2.rectangle(view_1_area, (gt_box_coords_vw_1[0], gt_box_coords_vw_1[1]),
        #               (gt_box_coords_vw_1[2], gt_box_coords_vw_1[3]), (colors[color_index]), 2)

        cv2.imshow("view_1_mark_area", view_1_area)
        # cv2.imshow("view_2_mark_area", view_2_area)
        # cv2.imwrite("view_1_area.jpg", view_1_area)
        # cv2.imwrite("ref_cam_frame_est.jpg", ref_cam_frame_est)
        cv2.imwrite("../performance_analysis/spatial_area_estimation_WT/cam_{}_{}_area_1.jpg".format(ref_cam, c_cam),
                    view_1_area)
        cv2.imwrite("../performance_analysis/spatial_area_estimation_WT/cam_{}_{}_area_1.jpg".format(c_cam, ref_cam),
                    view_2_area)
        cv2.waitKey(-1)

        # plot iou scores
        # plt.plot(iou_values_1, marker="o", label="reference_cam_iou", linewidth=3, markersize=10)
        # plt.xlabel("Total Analysed Frames", fontsize=18)
        # plt.ylabel("IoU Score of Overlap Area", fontsize=18)
        # plt.xticks(fontsize=16)
        # plt.yticks(fontsize=16)
        # plt.title("Iou Score vs No of Frames", fontsize=18)
        #
        # dump iou scores
        with open("iou_vs_fnumber_cam_1_{}.txt".format(c_cam), 'w') as iou_output:
            for iou in iou_values_1:
                iou_output.write("{}\n".format(iou))

        # break

    # cv2.waitKey(-1)
    # plt.show()
