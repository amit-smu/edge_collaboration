"""
module to generate priors for transformed (using coordiante mapper) ground truth of collaborative model
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
    model_file_path = "regression_models/lin_reg_deg_{}_src_{}_dst_{}_optimized".format(DEGREE, cam_pair[0],
                                                                                        cam_pair[2])
    print(model_file_path)
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
    img_dir = r"G:\Datasets\Wildtrack_dataset\PNGImages"
    label_dir = r"G:\Datasets\Wildtrack_dataset\labels_yolo"
    img_out_dir = "./temp"
    cam_pair = "7_5"  # transform one camera to another

    IOU_TH = 0.2
    DEGREE = 4
    width, height = (1920.0, 1080.0)
    resolutions = [1056, 512, 416, 320, 224, 128]
    spatial_overlap = {
        "4_1": [1089, 6, 1914, 1071],
        "7_5": [30, 57, 1480, 1042],
        "1_4": [6, 4, 908, 1074],
        "5_7": [51, 139, 1507, 1041]
    }

    x1, y1, x2, y2 = spatial_overlap[cam_pair]
    print("Cam pair: {}, Spatial Overlap : {}\n".format(cam_pair, spatial_overlap[cam_pair]))
    sh_width = x2 - x1
    sh_height = y2 - y1

    for i in range(1400, 2000, 5):
        label_name = "C{}_{:08d}.txt".format(cam_pair.split("_")[0], i)
        prior = np.full(shape=(int(height), int(width), 1), fill_value=0, dtype=np.uint8)

        with open("{}/{}".format(label_dir, label_name)) as label_file:
            content = label_file.readlines()

            for line in content:
                if len(line) <= 1:
                    continue
                line = line.strip()

                # read gt_1 for this image
                _, mid_x, mid_y, box_width, box_height = [x for x in line.split()]

                mid_x = float(mid_x) * width
                mid_y = float(mid_y) * height
                box_width = float(box_width) * width
                box_height = float(box_height) * height

                # enforce constraints
                # left = int(max(1, mid_x - box_width / 2))
                # top = int(max(1, mid_y - box_height / 2))
                # right = int(min(mid_x + box_width / 2, width - 1))
                # bottom = int(min(mid_y + box_height / 2, height - 1))

                left = int(mid_x - box_width / 2)
                top = int(mid_y - box_height / 2)
                right = int(mid_x + box_width / 2)
                bottom = int(mid_y + box_height / 2)

                #  check overlap with shared region
                if bb_icov(gt_box=[left, top, right, bottom], cropped_img_box=spatial_overlap[cam_pair]) >= IOU_TH:
                    mid_x = max(1, mid_x)
                    bt_y = max(1, mid_y + box_height / 2)

                    mid_x, bt_y, box_width, box_height = transform_coords(mid_x, bt_y, box_width, box_height)

                    left = int(max(1, mid_x - box_width / 2))
                    # top = int(max(1, mid_y - box_height / 2))
                    top = int(max(1, bt_y - box_height))
                    right = int(min(width - 1, mid_x + box_width / 2))
                    # bottom = int(min(height-1, mid_y + box_height / 2))
                    bottom = int(min(height - 1, bt_y))

                    collab_pair = cam_pair[::-1]  # reverse the string
                    if bb_icov(gt_box=[left, top, right, bottom],
                               cropped_img_box=spatial_overlap[collab_pair]) >= IOU_TH:
                        prior[top:bottom, left:right] = 250

            prior_name = "C{}_{:08d}_prior.png".format(cam_pair[2], i)
            print(prior_name)
            temp = spatial_overlap["7_5"]
            # cv2.rectangle(prior, (temp[0], temp[1]), (temp[2], temp[3]), (0, 0, 255), 2)
            cv2.imwrite("{}/{}".format(img_out_dir, prior_name), prior)
            # break
