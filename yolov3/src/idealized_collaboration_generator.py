"""
script used for idealized collaboration performance modelling (section 8.1 in paper).
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

    IOU_TH = 0.2
    img_width, img_height = (1920.0, 1080.0)
    resolutions = [1056, 512, 416, 320, 224, 128]
    RES = 416
    spatial_overlap = {
        "1_4": [6, 4, 908, 1074],
        "5_7": [51, 139, 1507, 1041]
    }

    x1, y1, x2, y2 = spatial_overlap[cam_pair]
    print("Cam pair: {}, Spatial Overlap : {}\n".format(cam_pair, spatial_overlap[cam_pair]))
    sh_width = x2 - x1
    sh_height = y2 - y1

    for i in range(1400, 2000, 5):
        name = "C{}_{:08d}".format(cam_pair[0], i)

        # extract shared region
        image_name = "{}.png".format(name)
        print(image_name)
        image = cv2.imread("{}/{}".format(img_dir, image_name))
        assert image is not None
        sh_region = image[y1:y2, x1:x2]

        # cv2.imshow("image", image)
        # cv2.imshow("sh_reg", sh_region)
        # cv2.waitKey(-1)

        sh_width_new = int((sh_width / img_width) * RES)
        sh_height_new = int((sh_height / img_height) * RES)
        temp = cv2.resize(sh_region, dsize=(sh_width_new, sh_height_new), interpolation=cv2.INTER_AREA)
        sh_region = cv2.resize(temp, dsize=(sh_width, sh_height), interpolation=cv2.INTER_CUBIC)
        # image[:, :, :] = 0
        image[y1:y2, x1:x2] = sh_region
        sh_region = image[y1 - 1:y2 + 1, x1 - 1:x2 + 1]

        # cv2.imshow("new_img", image)
        # cv2.waitKey(-1)
        # add border to image (like a padding)
        image = cv2.copyMakeBorder(sh_region, y1 - 1, int(img_height - y2 - 1), x1 - 1, int(img_width - x2 - 1),
                                   borderType=cv2.BORDER_REPLICATE,
                                   value=255)
        # cv2.copyMakeBorder()
        # print(image.shape)

        cv2.imwrite("{}/{}".format(img_out_dir, image_name), image)

        label_name = "{}.txt".format(name)
        # print(label_name)
        prior = np.full(shape=(int(img_height), int(img_width), 1), fill_value=0, dtype=np.uint8)
        with open("{}/{}".format(label_dir, label_name)) as label_file:
            content = label_file.readlines()

            for line in content:
                if len(line) <= 1:
                    continue
                line = line.strip()

                # read gt for this image
                _, mid_x, mid_y, box_width, box_height = [x for x in line.split()]

                mid_x = float(mid_x) * img_width
                mid_y = float(mid_y) * img_height
                box_width = float(box_width) * img_width
                box_height = float(box_height) * img_height

                # enforce constraints
                left = int(max(1, mid_x - box_width / 2))
                top = int(max(1, mid_y - box_height / 2))
                right = int(min(mid_x + box_width / 2, img_width - 1))
                bottom = int(min(mid_y + box_height / 2, img_height - 1))

                #  check overlap with shared region
                if bb_icov(gt_box=[left, top, right, bottom], cropped_img_box=spatial_overlap[cam_pair]) >= IOU_TH:
                    prior[top:bottom, left:right] = 250
        prior_subset = prior[y1:y2, x1:x2]
        cv2.imwrite("{}/{}_prior.png".format(img_out_dir, label_name[:-4]), prior)
        # break
