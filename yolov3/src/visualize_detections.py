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

            mid_x = float(line[1]) * 1920
            mid_y = float(line[2]) * 1080
            width = float(line[3]) * 1920
            height = float(line[4]) * 1080

            xmin = int(mid_x - (width / 2))
            xmax = int(mid_x + (width / 2))
            ymin = int(mid_y - (height / 2))
            ymax = int(mid_y + (height / 2))
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), thickness=2)


if __name__ == "__main__":
    labels_dir =r"G:\Datasets\Wildtrack_dataset\labels"
    img_dir = r"G:\Datasets\Wildtrack_dataset\PNGImages"
    result_file = "result.txt"

    SEPARATOR_KEY = "Enter Image Path:"
    IMG_FORMAT = ".png"
    img_width, img_height = (1920, 1080)

    with open(result_file) as detections:
        detections = detections.readlines()
        image_name = None
        for line in detections:
            if SEPARATOR_KEY in line:
                image_path = re.search(SEPARATOR_KEY + '(.*)' + IMG_FORMAT, line)
                # get the image name (the final component of a image_path)
                # e.g., from 'data/horses_1' to 'horses_1'
                if image_name is not None:
                    cv2.imwrite("{}/{}.png".format("./temp/", image_name), image)
                    # break
                image_name = os.path.basename(image_path.group(1))
                image = cv2.imread("{}/{}.png".format(img_dir, image_name))
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
