"""
generate prior from transformed detected boxes from collaborating camera
"""
import cv2
import numpy as np
import pickle
from sklearn.preprocessing import PolynomialFeatures


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


def load_regression_model():
    model = None
    model_file_path = reg_model_path.format(DEGREE, cam_pair[0], cam_pair[2])
    # print(model_file_path)
    # print("degree: {}, src_cam: {}\n".format(degree, src_cam))
    with open(model_file_path, 'rb') as input_file:
        model = pickle.load(input_file)
    return model


def transform_coords(mid_x, bt_y, bw, bh):
    reg_model = load_regression_model()
    poly_features = PolynomialFeatures(degree=DEGREE, interaction_only=False)

    X = np.array([mid_x, bt_y, bw, bh], dtype=np.int).reshape((1, -1))
    X_poly = poly_features.fit_transform(X)
    y_pred = reg_model.predict(X_poly)
    return y_pred[0]


if __name__ == "__main__":
    img_dir = r"../../../../rpi_hardware/raw_image_processing/data/episode_1/pi_{}_frames_1056_v2"
    label_dir = r"./collaborating_cam_detected_objs_mAP_repo_tr_gt"
    img_out_dir = "./data"
    frame_numbers_path = "./test_frame_numbers.txt"
    reg_model_path = r"../coordinate_mapping/regression_models/lin_reg_deg_{}_src_{}_dst_{}_optimized"

    DEGREE = 4
    ref_cam = 2
    collab_cam = 1
    cam_pair = "{}_{}".format(collab_cam, ref_cam)
    VISUALIZE = False

    IOU_TH = 0.2
    width, height = (1056.0, 1056.0)
    spatial_overlap = {
        "2_1": [360, 0, 1010, 1080],
        "2_3": [0, 0, 470, 1080],
        "1_2": [0, 0, 480, 1080],
        "3_2": []
    }
    scale = 1056 / 1080
    spatial_overlap[cam_pair] = [int(d * scale) for d in spatial_overlap[cam_pair]]

    x1, y1, x2, y2 = spatial_overlap[cam_pair]
    print("Cam pair: {}, Spatial Overlap : {}, Degree: {}\n".format(cam_pair, spatial_overlap[cam_pair], DEGREE))

    sh_width = x2 - x1
    sh_height = y2 - y1

    with open(frame_numbers_path) as in_file:
        frame_numbers = in_file.read().split("\n")
    for number in frame_numbers:
        if len(number) == 0:
            continue
        label_name = "frame_{}_{}.txt".format(collab_cam, number)
        print(label_name)

        if VISUALIZE:
            image = cv2.imread("{}/frame_{}_{}.jpg".format(img_dir.format(ref_cam), ref_cam, number))
            assert image is not None
            cam_pair_reversed = cam_pair[::-1]
            x1_r, y1_r, x2_r, y2_r = spatial_overlap[cam_pair_reversed]
            cv2.rectangle(image, (x1_r, y1_r), (x2_r, y2_r), (0, 0, 255), 2)

        with open("{}/{}".format(label_dir, label_name)) as label_file:
            content = label_file.readlines()
            prior = np.full(shape=(int(height), int(width), 1), fill_value=0, dtype=np.uint8)
            for line in content:
                if len(line) <= 1:
                    continue
                line = line.strip()

                # read detected boxes for this image
                left, top, right, bottom = [int(x) for x in line.split()[2:]]
                #  check overlap with shared region
                if bb_icov(gt_box=[left, top, right, bottom], cropped_img_box=spatial_overlap[cam_pair]) >= IOU_TH:
                    box_width = right - left
                    box_height = bottom - top
                    bt_mid_x = int(max(1, left + box_width / 2))
                    bt_mid_x, bottom, box_width, box_height = transform_coords(bt_mid_x, bottom, box_width, box_height)
                    left = int(max(1, bt_mid_x - box_width / 2))
                    top = int(max(1, bottom - box_height))
                    right = int(min(width - 1, bt_mid_x + box_width / 2))
                    bottom = int(min(height - 1, bottom))

                    collab_pair = cam_pair[::-1]  # reverse the string
                    if bb_icov(gt_box=[left, top, right, bottom],
                               cropped_img_box=spatial_overlap[collab_pair]) >= IOU_TH:
                        prior[top:bottom, left:right] = 255

                    if VISUALIZE:
                        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0))

                # write prior to file
            out_prior_name = "{}/frame_{}_{}_prior.jpg".format(img_out_dir, ref_cam, number)
            # print(out_prior_name)
            cv2.imwrite(out_prior_name, prior)
            if VISUALIZE:
                # cv2.imwrite("{}/{}.png".format(img_out_dir, label_name[:-4]), image)
                image_copy = cv2.resize(image, (700, 700))
                cv2.imshow("img", image_copy)
                prior_copy = cv2.resize(prior, (700, 700))
                cv2.imshow("prior", prior_copy)
                cv2.waitKey(-1)
