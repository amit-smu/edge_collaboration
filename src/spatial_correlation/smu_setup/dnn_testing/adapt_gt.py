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
    img_dir = r"../../../../rpi_hardware/raw_image_processing/data/episode_1/pi_2_frames_1056_v2"
    label_dir = r"../../../../rpi_hardware/raw_image_processing/data/episode_1/ground_truth/frame_wise_gt_yolo/cam_2"
    img_out_dir = "./data"
    cam_pair = "2_1"
    frame_numbers_path = "./test_frame_numbers.txt"
    ref_cam = 2
    VISUALIZE = False

    IOU_TH = 0.2
    width, height = (1056.0, 1056.0)
    spatial_overlap = {
        "2_1": [360, 0, 1010, 1080],
        "2_3": [0, 0, 470, 1080]
    }
    scale = 1056 / 1080
    spatial_overlap[cam_pair] = [int(d * scale) for d in spatial_overlap[cam_pair]]

    x1, y1, x2, y2 = spatial_overlap[cam_pair]
    print("Cam pair: {}, Spatial Overlap : {}\n".format(cam_pair, spatial_overlap[cam_pair]))

    sh_width = x2 - x1
    sh_height = y2 - y1

    with open(frame_numbers_path) as in_file:
        frame_numbers = in_file.read().split("\n")
    # for i in range(1400, 2000, 5):
    for number in frame_numbers:
        if len(number) == 0:
            continue
        label_name = "frame_{}_{}.txt".format(ref_cam, number)
        print(label_name)

        if VISUALIZE:
            image = cv2.imread("{}/{}.jpg".format(img_dir, label_name[:-4]))
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

        with open("{}/{}".format(label_dir, label_name)) as label_file:
            content = label_file.readlines()
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
                        if VISUALIZE:
                            cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0))
                if VISUALIZE:
                    # cv2.imwrite("{}/{}.png".format(img_out_dir, label_name[:-4]), image)
                    image_copy = cv2.resize(image, (700, 700))
                    cv2.imshow("img", image_copy)
                    cv2.waitKey(-1)