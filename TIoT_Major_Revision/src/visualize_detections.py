import os
import cv2


def load_class_names(names_file):
    with open(names_file, 'r') as f:
        return [line.strip() for line in f.readlines()]


if __name__ == "__main__":
    INPUT_PATH = "../dataset/wildtrack/Images"
    image_names = os.listdir(INPUT_PATH)

    classes = load_class_names("custom.names")
    name_to_index = {i: str(name) for i, name in enumerate(classes)}

    for name in image_names:
        if name.__contains__("png"):
            print(name)
            image = cv2.imread(f"{INPUT_PATH}/{name}")

            gt_file = f"{name[:-4]}.txt"
            with open(f"{INPUT_PATH}/{gt_file}") as result_file:
                detections = result_file.readlines()
                for obj in detections:
                    obj_class, obj_coordinates = obj.strip().split()[0], obj.strip().split()[1:]
                    obj_class = name_to_index[int(obj_class)]

                    x, y, w, h = [int(float(o) * 1056) for o in obj_coordinates]

                    cv2.rectangle(image, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)),
                                  (255, 0, 0))
                    cv2.putText(image, obj_class, (int(x - w / 2), int(y - h / 2)), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1,
                                color=(255, 0, 255))
            cv2.imshow("image", image)
            cv2.waitKey(-1)
