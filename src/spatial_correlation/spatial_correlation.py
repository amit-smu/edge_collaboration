"""
main module
module to establish spatial overlap between pair of cameras
"""

import cv2
import pandas as pd
import utils
import numpy as np
from src import obj_comp_utils as comp_utils
from matplotlib import pyplot as plt
import xml.dom.minidom
import random
from sklearn import svm

# global variables
DEBUG = False
THR_color_hist = 0.2
THR_sift = 0.3


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
    if INCLUDE_COLOR:
        features['color'] = utils.compute_color_hist(img_obj)
    if INCLUDE_SIFT:
        features['sift'] = utils.compute_sift_features(img_obj)
    if INCLUDE_EMBEDDING:
        features['embedding'] = utils.get_obj_embedding(img_obj, temp_dir_path="../temp_files")
    return features


def get_scores(obj1, obj2):
    if INCLUDE_COLOR:
        color_score = utils.compare_color_hist(obj1, obj2)
    else:
        color_score = None

    if INCLUDE_SIFT:
        sift_score = utils.compare_sift(obj1, obj2)
    else:
        sift_score = None

    if INCLUDE_EMBEDDING:
        eucl_distance = utils.get_eucl_distance(obj1['embedding'], obj2['embedding'])
    else:
        eucl_distance = None

    return color_score, sift_score, eucl_distance


def compare_objects(obj1, obj2):
    """
    compare if two objects are same or not based on scores from various object matching method scores.
    :param obj1:
    :param obj2:
    :return:
    """
    objects_same = False
    color_score, sift_score, eucl_distance = get_scores(obj1, obj2)

    # logic for objects being same
    if color_score <= THR_color_hist or sift_score >= THR_sift:
        objects_same = True
    return objects_same


def get_min_max_coords(coordinates):
    min_x_list = [x[0][0] for x in coordinates]
    min_x = min(min_x_list)
    max_x_list = [x[1][0] for x in coordinates]
    max_x = max(max_x_list)
    min_y_list = [x[0][1] for x in coordinates]
    min_y = min(min_y_list)
    max_y_list = [x[1][1] for x in coordinates]
    max_y = max(max_y_list)
    return min_x, max_x, min_y, max_y


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


def get_smallest_polygon(coords):
    """
    for a given list of points coordinates, get four points that represent smallest POLYGON fitting all the given points
    :param coords:
    :return:
    """
    print("")


def draw_rect(coords, img, color):
    """
    draw rectangle with given points
    :param coords:
    :param img:
    :return:
    """
    cv2.rectangle(img, (coords[0], coords[1]), (coords[2], coords[3]), color, 2)


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
    sp_overlap_dir = "../../dataset/spatial_overlap/PETS"
    xml_path = "{}/{}".format(sp_overlap_dir, xml_file_name)
    return get_bb_coords(xml_path=xml_path)


def match_objects(objects_1, objects_2):
    """
    matches objects in one set to teh objects in other set, finds best possible match using the compare_objects method
    :param objects_1:
    :param objects_2:
    :return:
    """
    object_mappings = []

    for obj1 in objects_1:
        for obj2 in objects_2:
            obj_same = utils.same_objects(obj1['embedding'], obj2['embedding'])
            if obj_same:
                # add it to dataframe
                object_mappings.append(
                    [obj1['rand_id'], obj2['rand_id'], 1.0, obj1['location'], obj2['location']])

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


def weighted_spatial_overlap_performance(masked_image, gt_overlap_area):
    """
    how does estimated overlap IoU varies with various values of pixel intensity threshold
    :param masked_image: The estimated overlap area
    :param gt_overlap_area: coordinates of ground truth of overlap area
    :return:
    """
    print()
    pixel_intensities = [245, 230, 210, 190, 170, 150, 130, 110, 90, 70, 50, 30, 20, 10, 0]
    iou_scores = []  # iou score for each pixel intensity

    masked_image_org = masked_image.copy()

    masked_image = masked_image[:, :, 0]  # drop 3rd dimension(depth)
    image_shape = masked_image.shape
    for intensity_threshold in pixel_intensities:
        desired_coordinates = []
        # select all pixels below this intensity value
        for r, c in np.ndindex(image_shape):
            if masked_image[r, c] <= intensity_threshold:
                desired_coordinates.append((c, r))  # x,y

        # estimate enclosing rectangle for all these points
        min_x = min(desired_coordinates, key=lambda x: x[0])[0]
        min_y = min(desired_coordinates, key=lambda x: x[1])[1]
        max_x = max(desired_coordinates, key=lambda x: x[0])[0]
        max_y = max(desired_coordinates, key=lambda x: x[1])[1]
        iou_score = utils.bb_iou([min_x, min_y, max_x, max_y], gt_overlap_area)
        iou_scores.append(iou_score)

        if DEBUG:
            masked_image_copy = masked_image_org.copy()
            cv2.rectangle(masked_image_copy, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)
            cv2.imshow("masked_image_{}".format(intensity_threshold), masked_image_copy)
            cv2.imshow("masked_image", masked_image)
            cv2.waitKey(15)
    plt.plot(pixel_intensities, iou_scores)
    plt.show()
    print(iou_scores)
    print(pixel_intensities)
    return iou_scores, pixel_intensities


if __name__ == "__main__":
    dir_name = "../../dataset/"
    view_1_name = "View_007"
    view_2_name = "View_008"
    EUCL_THRESHOLD = 12
    image_size = (720, 576)  # PETS dataset (width, height)

    SAME_OBJECTS_ANALYSIS = False  # whether to do same obj analysis or not
    INCLUDE_COLOR = False
    INCLUDE_SIFT = False
    INCLUDE_EMBEDDING = True

    # create white images (used for marking the mapped areas)
    view_1_area = np.full((image_size[1], image_size[0]), 255, dtype=np.uint8)
    view_2_area = np.full((image_size[1], image_size[0]), 255, dtype=np.uint8)
    view_1_area_copy = view_1_area.copy()
    view_2_area_copy = view_2_area.copy()

    # ground truth spatial overlap area coordinates
    gt_box_coords_vw_1 = get_gt_sp_overlap_coordinates("07", "08")
    gt_box_coords_vw_2 = get_gt_sp_overlap_coordinates("08", "07")

    # estimated spatial overlap area coordinates
    est_box_coords_vw1_global = []
    est_box_coords_vw2_global = []

    # iou values for most recent estimated spatial overlap frames
    iou_values_1 = []
    iou_values_2 = []

    view_1_grnd_truth = pd.read_csv("{}/{}.txt".format(dir_name, view_1_name), delimiter=" ")
    view_2_grnd_truth = pd.read_csv("{}/{}.txt".format(dir_name, view_2_name), delimiter=" ")

    # frame_number = 1
    for frame_number in range(0, 635):  # 80% training data
        print("frame number : {}".format(frame_number))
        frame_name = "frame_{:04d}".format(frame_number)

        view_1_frame = cv2.imread("{}/{}/{}.jpg".format(dir_name, view_1_name, frame_name))
        view_2_frame = cv2.imread("{}/{}/{}.jpg".format(dir_name, view_2_name, frame_name))

        # get objects from frames (using ground truth)
        view_1_objects = view_1_grnd_truth[view_1_grnd_truth['frame_number'] == frame_number]
        view_2_objects = view_2_grnd_truth[view_2_grnd_truth['frame_number'] == frame_number]
        # apply filters on objects
        view_1_objects = view_1_objects[(view_1_objects['lost'] == 0) & (view_1_objects['occluded'] == 0)]
        view_2_objects = view_2_objects[(view_2_objects['lost'] == 0) & (view_2_objects['occluded'] == 0)]

        # compute object features for each object in frame
        view_1_obj_features = []
        view_2_obj_features = []
        view_1_obj_features = get_all_objects_features(view_1_objects, view_1_frame)
        view_2_obj_features = get_all_objects_features(view_2_objects, view_2_frame)

        # compare objects from view 1 with objects from view 2
        vw_1_matching_obj_coord = []
        vw_2_matching_obj_coord = []

        obj_mappings = match_objects(view_1_obj_features, view_2_obj_features)
        if obj_mappings is None:
            continue
        # mark overlapping areas
        mark_matching_obj_area(obj_mappings=obj_mappings, view_1_area=view_1_area, view_2_area=view_2_area,
                               view_1_obj_features=view_1_obj_features, view_2_obj_features=view_2_obj_features)

        cv2.imwrite("intermediate_frames/marked_area_cam_7_f_{}.jpg".format(frame_number), view_1_area)
        cv2.imwrite("intermediate_frames/marked_area_cam_8_f_{}.jpg".format(frame_number), view_2_area)

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

        est_box_coords_vw1 = get_fitting_rect_coordinates(view_1_obj_features, dataframe=obj_mappings,
                                                          column="obj_id_1")
        # min_x_2, max_x_2, min_y_2, max_y_2 = get_min_max_coords(vw_2_matching_obj_coord)
        est_box_coords_vw2 = get_fitting_rect_coordinates(view_2_obj_features, dataframe=obj_mappings,
                                                          column="obj_id_2")

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
            draw_rect(gt_box_coords_vw_1, view_1_frame, (0, 255, 0))
            cv2.putText(view_1_frame, "ground_truth", (gt_box_coords_vw_1[0], gt_box_coords_vw_1[1] - 10),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, (0, 255, 0), 1)
            draw_rect(est_box_coords_vw1_global, view_1_frame, (0, 0, 255))
            cv2.putText(view_1_frame, "estimated overlap",
                        (est_box_coords_vw1_global[0], est_box_coords_vw1_global[1] - 10),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, (0, 0, 255), 1)
            cv2.imshow("Camera_1", view_1_frame)

            draw_rect(gt_box_coords_vw_2, view_2_frame, (0, 255, 0))
            cv2.putText(view_2_frame, "ground_truth", (gt_box_coords_vw_2[0], gt_box_coords_vw_2[1] - 10),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, (0, 255, 0), 1)
            draw_rect(est_box_coords_vw2_global, view_2_frame, (0, 0, 255))
            cv2.putText(view_2_frame, "estimated overlap",
                        (est_box_coords_vw2_global[0], est_box_coords_vw2_global[1] - 10),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, (0, 0, 255), 1)
            cv2.imshow("Camera_2", view_2_frame)

            cv2.waitKey(1)

    # plot number of frames vs iou of estimated spatial overlap

    # save the masked frames from both camera views
    cv2.imwrite("view_1_area.jpg", view_1_area)
    cv2.imwrite("view_2_area.jpg", view_2_area)

    x_axis = np.arange(1, len(iou_values_1) + 1)
    y_axis = iou_values_1
    # plt.line(x_axis, y_axis)
    plt.plot(y_axis, marker="o", label="reference_cam_iou", linewidth=3, markersize=10)
    plt.xlabel("Total Analysed Frames", fontsize=18)
    plt.ylabel("IoU Score of Overlap Area", fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.title("Camera 7", fontsize=18)

    plt.figure()
    x_axis = np.arange(1, len(iou_values_2) + 1)
    y_axis = iou_values_2
    # plt.line(x_axis, y_axis)
    plt.plot(y_axis, marker="o", label="reference_cam_iou", linewidth=3, markersize=10)
    plt.xlabel("Total Analysed Frames", fontsize=18)
    plt.ylabel("IoU Score of Overlap Area", fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.title("Camera 8", fontsize=18)

    with open("camera_7_iou_vs_frames.txt", 'w') as cam7:
        for value in iou_values_1:
            cam7.write("{}\n".format(value))
    with open("camera_8_iou_vs_frames.txt", 'w') as cam8:
        for value in iou_values_2:
            cam8.write("{}\n".format(value))

    # plt.show()
    # cv2.waitKey(-1)
