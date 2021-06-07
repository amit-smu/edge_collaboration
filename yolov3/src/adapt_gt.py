"""
module to change ground truth for "shared-region only" study
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


if __name__ == "__main__":
    img_dir = r"G:\Datasets\Wildtrack_dataset\PNGImages"
    label_dir = r"G:\Datasets\Wildtrack_dataset\labels_yolo"
    img_out_dir = "./temp"
    cam_pair = "1_4"

    IOU_TH = 0.75
    width, height = (1920.0, 1080.0)
    resolutions = [1056, 512, 416, 320, 224, 128]
    spatial_overlap = {
        "1_4": [6, 4, 908, 1074],
        "5_7": [51, 139, 1507, 1041]
    }

    x1, y1, x2, y2 = spatial_overlap[cam_pair]
    print("Cam pair: {}, Spatial Overlap : {}\n".format(cam_pair, spatial_overlap[cam_pair]))
    sh_width = x2 - x1
    sh_height = y2 - y1

    for i in range(1400, 2000, 5):
        label_name = "C{}_{:08d}.txt".format(cam_pair[0], i)
        print(label_name)
        # image = cv2.imread("{}/{}.png".format(img_dir, label_name[:-4]))
        with open("{}/{}".format(label_dir, label_name)) as label_file:
            content = label_file.readlines()
            prior = np.full(shape=(int(height), int(width), 1), fill_value=0, dtype=np.uint8)

            with open("{}/{}".format(img_out_dir, label_name), 'w') as out_label_file:
                for line in content:
                    if len(line) <= 1:
                        continue
                    line_copy = line
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
                    if bb_icov(gt_box=[left, top, right, bottom], cropped_img_box=spatial_overlap[cam_pair]) >= IOU_TH:
                        out_label_file.write("{}".format(line_copy))
                        # realign coordinates
                        # left = max(1, left - x1)
                        # top = max(1, top - y1)
                        # right = min(sh_width - 1, right - x1)
                        # bottom = min(sh_height - 1, bottom - y1)
                        #
                        # # convert to yolo format
                        # bx_width = right - left
                        # bx_height = bottom - top
                        # bx_mid_x = left + bx_width / 2
                        # bx_mid_y = top + bx_height / 2
                        #
                        # bx_width = float(bx_width) / sh_width
                        # bx_height = float(bx_height) / sh_height
                        # bx_mid_x = float(bx_mid_x) / sh_width
                        # bx_mid_y = float(bx_mid_y) / sh_height
                        # out_label_file.write("0 {} {} {} {}\n".format(bx_mid_x, bx_mid_y, bx_width, bx_height))
        # break
