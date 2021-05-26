import cv2
import os

if __name__ == "__main__":
    in_dir = r"D:/My Drive/yolov3/prior_img_model/data/Images"
    annot_dir = r"D:/My Drive/yolov3/prior_img_model/data/Images"

    filenames = os.listdir(in_dir)
    for name in filenames:
        if not name.__contains__("png"):
            continue
        if name.__contains__("prior"):
            continue
        img = cv2.imread("{}/{}".format(in_dir, name), cv2.IMREAD_UNCHANGED)
        with open("{}/{}.txt".format(annot_dir, name[:-4]), 'r') as annot:
            line = annot.readline().strip("\n").split(" ")
            while len(line) > 2:
                mid_x = float(line[1]) * 1056
                mid_y = float(line[2]) * 1056
                width = float(line[3]) * 1056
                height = float(line[4]) * 1056

                xmin = int(mid_x - (width / 2))
                xmax = int(mid_x + (width / 2))
                ymin = int(mid_y - (height / 2))
                ymax = int(mid_y + (height / 2))
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), thickness=2)
                line = annot.readline().strip("\n").split(" ")

        print(img.shape)
        cv2.imshow("{}".format(name), img[:, :, 0:3])
        cv2.waitKey(-1)
        cv2.destroyAllWindows()
