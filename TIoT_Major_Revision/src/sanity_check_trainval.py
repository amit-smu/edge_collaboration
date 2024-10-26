import os
import sys
import cv2


def process_gt(line):
    class_id = line[0]
    line = [float(l) for l in line.strip().split()[1:]]

    c_x, c_y, w, h = line
    x1 = (c_x * 1056) - (w * 1056) / 2
    y1 = (c_y * 1056) - (h * 1056) / 2
    x2 = x1 + (w * 1056)
    y2 = y1 + (h * 1056)
    return class_id, int(x1), int(y1), int(x2), int(y2)


if __name__ == "__main__":
    input_file = "train.txt"
    tmp_dir = "./temp"
    id_to_name = {
        "0": "Person",
        "1": "Car"
    }

    with open(input_file, 'r') as f:
        data = f.readlines()

        for name in data:
            name = name.strip()[3:]
            print(name)
            image = cv2.imread(name)
            assert image is not None

            gt_file = f"{name[:-4]}.txt"
            with open(gt_file, 'r') as gt:
                gt_data = gt.readlines()
                for d in gt_data:
                    class_id, x1, y1, x2, y2 = process_gt(d.strip())
                    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(image, id_to_name[class_id], (x1, y1), fontFace=cv2.FONT_HERSHEY_PLAIN,
                                fontScale=2, color=(255, 0, 0), thickness=2)

                file_name = name.split("/")[-1]
                cv2.imwrite(f"{tmp_dir}/{file_name}", image)
                # cv2.waitKey(-1)
                # cv2.destroyAllWindows()

                break
