"""
module to generate priors for transformed (using coordiante mapper) ground truth of collaborative model
"""
import cv2
import numpy as np


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


def transform_coords(x1, y1, x2, y2):
    print()


if __name__ == "__main__":
    img_dir = r"G:\Datasets\Wildtrack_dataset\PNGImages"
    label_dir = r"G:\Datasets\Wildtrack_dataset\labels_yolo"
    img_out_dir = "./temp"
    cam_pair = "7_5"

    IOU_TH = 0.5
    width, height = (1920.0, 1080.0)
    resolutions = [1056, 512, 416, 320, 224, 128]
    spatial_overlap = {
        "4_1": [1089, 6, 1914, 1071],
        "7_5": [30, 57, 1480, 1042]  ######### update this ######
    }

    x1, y1, x2, y2 = spatial_overlap[cam_pair]
    print("Cam pair: {}, Spatial Overlap : {}\n".format(cam_pair, spatial_overlap[cam_pair]))
    sh_width = x2 - x1
    sh_height = y2 - y1

    for i in range(1400, 2000, 5):
        label_name = "C{}_{:08d}.txt".format(cam_pair.split("_")[0], i)
        print(label_name)

        with open("{}/{}".format(label_dir, label_name)) as label_file:
            content = label_file.readlines()
            prior = np.full(shape=(int(height), int(width), 1), fill_value=0, dtype=np.uint8)

            for line in content:
                if len(line) <= 1:
                    continue
                line = line.strip()

                # read gt for this image
                _, mid_x, mid_y, box_width, box_height = [x for x in line.split()]

                mid_x = float(mid_x) * width
                mid_y = float(mid_y) * height
                box_width = float(box_width) * width
                box_height = float(box_height) * height

                # enforce constraints
                left = int(max(1, mid_x - box_width / 2))
                top = int(max(1, mid_y - box_height / 2))
                right = int(min(mid_x + box_width / 2, width - 1))
                bottom = int(min(mid_y + box_height / 2, height - 1))

                #  check overlap with shared region
                if bb_icov(gt_box=[left, top, right, bottom], cropped_img_box=spatial_overlap[cam_pair]) >= 0.5:
                    left, top, right, bottom = transform_coords([left, top, right, bottom])
                    prior[top:bottom, left:right] = 250
