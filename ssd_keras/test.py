import cv2
import os
from xml.etree import ElementTree

wt_image_dir = r"D:\GitRepo\edge_computing\edge_collaboration\dataset\Wildtrack_dataset\PNGImages"
wt_annot_dir = r"D:\GitRepo\edge_computing\edge_collaboration\dataset\Wildtrack_dataset\Annotations"
all_images = os.listdir(wt_image_dir)

for img_name in all_images:
    if img_name.__contains__("C4"):
        image_1 = cv2.imread("{}/{}".format(wt_image_dir, "C1_{}".format(img_name[3:])))
        image_4 = cv2.imread("{}/{}".format(wt_image_dir, "C4_{}".format(img_name[3:])))

        assert image_1 is not None
        assert image_4 is not None

        cv2.rectangle(image_1, (6, 4), (828, 1074), (0, 255, 0), 1)
        cv2.rectangle(image_4, (1089, 6), (1914, 1071), (0, 255, 0), 1)

        # draw gt boxes image 1
        annot_name_1 = "C1_{}.xml".format(img_name[3:-4])
        annot_file_path = "{}/{}".format(wt_annot_dir, annot_name_1)
        root = ElementTree.parse(annot_file_path).getroot()
        assert root is not None
        objects = root.findall("object")
        for obj in objects:
            bndbox = obj.find("bndbox")
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)
            cv2.rectangle(image_1, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)

        annot_name_4 = "C4_{}.xml".format(img_name[3:-4])
        annot_file_path = "{}/{}".format(wt_annot_dir, annot_name_4)
        root = ElementTree.parse(annot_file_path).getroot()
        assert root is not None
        objects = root.findall("object")
        for obj in objects:
            bndbox = obj.find("bndbox")
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)

            cv2.rectangle(image_4, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)

        cv2.imshow("image_1", image_1[4:1074, 6:828])
        cv2.imshow("image_4", image_4[6:1071, 1089:1914])
        cv2.waitKey(-1)
