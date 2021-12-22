"""
module to generate shared regions of a specific size and resolution.
Script is used for study in section 6 of paper
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
    cam_pair = "5_7"
    position = "66_right"

    IOU_TH = 0.2
    img_width, img_height = (1920.0, 1080.0)
    resolutions = [1056, 512, 416, 224, 128]
    res = 1056
    spatial_overlap = {
        "33_left": [0, 0, 640, 1080],
        "33_right": [1280, 0, 1920, 1080],
        "66_left": [0, 0, 1280, 1080],
        "66_right": [640, 0, 1920, 1080]
    }

    x1, y1, x2, y2 = spatial_overlap[position]
    print("Cam pair: {}, Spatial Overlap : {}\n".format(cam_pair, spatial_overlap[position]))
    sh_width = x2 - x1
    sh_height = y2 - y1

    for i in range(1400, 2000, 5):
        name = "C{}_{:08d}".format(cam_pair[0], i)
        image_name = "{}.png".format(name)
        print(image_name)
        image = cv2.imread("{}/{}".format(img_dir, image_name))
        assert image is not None

        # extract shared region
        sh_region = image[y1:y2, x1:x2]
        sr_width_new = int((sh_width / img_width) * res)
        sr_height_new = int((sh_height / img_height) * res)
        temp = cv2.resize(sh_region, dsize=(sr_width_new, sr_height_new), interpolation=cv2.INTER_AREA)
        sh_region = cv2.resize(temp, dsize=(sh_width, sh_height), interpolation=cv2.INTER_CUBIC)
        image[y1:y2, x1:x2] = sh_region
        cv2.imwrite("{}/{}".format(img_out_dir, image_name), image)

        # create prior
        label_name = "{}.txt".format(name)
        # print(label_name)
        prior = np.full(shape=(int(img_height), int(img_width), 1), fill_value=0, dtype=np.uint8)
        with open("{}/{}".format(label_dir, label_name)) as label_file:
            content = label_file.readlines()
            for line in content:
                line = line.strip()
                if len(line) <= 1:
                    continue

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
                if bb_icov(gt_box=[left, top, right, bottom], cropped_img_box=spatial_overlap[position]) >= IOU_TH:
                    prior[top:bottom, left:right] = 255

        # write prior
        cv2.imwrite("{}/{}_prior.png".format(img_out_dir, image_name[:-4]), prior)
        # break
