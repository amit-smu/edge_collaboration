"""
given a list of images and detections, this module draws bboxes on images

"""
import os
import re
import cv2


def draw_gt(image, image_name):
    with open("{}/{}.txt".format(labels_dir, image_name)) as gt_file:
        lines = gt_file.readlines()
        for line in lines:
            if len(line) == 1:
                continue
            line = line.strip()
            line = line.split()
            # mid_x = float(mid_x)
            # mid_y = float(mid_y)
            # width = float(width)
            # height = float(height)

            mid_x = float(line[1]) * img_width
            mid_y = float(line[2]) * img_height
            width = float(line[3]) * img_width
            height = float(line[4]) * img_height

            xmin = int(mid_x - (width / 2))
            xmax = int(mid_x + (width / 2))
            ymin = int(mid_y - (height / 2))
            ymax = int(mid_y + (height / 2))
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), thickness=2)


if __name__ == "__main__":
    labels_dir =r"../../../../rpi_hardware/raw_image_processing/data/episode_1/ground_truth/frame_wise_gt_yolo/cam_2"
    img_dir = r"../../../../rpi_hardware/raw_image_processing/data/episode_1/pi_2_frames_1056_v2"
    result_file = r"./result_1056.txt"

    SEPARATOR_KEY = "Enter Image Path:"
    IMG_FORMAT = ".jpg"
    img_width, img_height = (1056, 1056)

    with open(result_file) as detections:
        detections = detections.readlines()
        image_name = None
        for line in detections:
            if SEPARATOR_KEY in line:
                image_path = re.search(SEPARATOR_KEY + '(.*)' + IMG_FORMAT, line)
                # get the image name (the final component of a image_path)
                # e.g., from 'data/horses_1' to 'horses_1'
                if image_name is not None:
                    # cv2.imwrite("{}/{}.png".format("./detection_temp/", image_name), image)
                    cv2.imshow("image", image)
                    cv2.waitKey(-1)
                    # break
                image_name = os.path.basename(image_path.group(1))
                image = cv2.imread("{}/{}.jpg".format(img_dir, image_name))
                assert image is not None
                draw_gt(image, image_name)
                print(image_name)
            else:
                # draw bboxes on image
                line = line.strip()
                bbox = line.split(", ")[1]
                x, y, w, h = bbox.split()
                x = int(x)
                y = int(y)
                w = int(w)
                h = int(h)
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
