"""
the images from cityflow v2 dataset are in 1920x1080 resolution, but our Yolo model takes 1056x1056
"""
import cv2
import os

if __name__=="__main__":
    INPUT_PATH = "../dataset/S03/c015"
    image_names = os.listdir(INPUT_PATH)
    assert image_names is not None

    print(image_names)
    count = 0
    for name in image_names:
        print(name)
        if not name.__contains__("jpg"):
            continue
        img_path = "{}/{}".format(INPUT_PATH, name)
        image = cv2.imread(img_path)
        image = image[12:1068, 432:1488]
        # print(image.shape)
        # image = cv2.resize(image, dsize=(1056,1056))
        # cv2.imshow("image", image)
        cv2.imwrite(filename=img_path, img= image)
        # cv2.waitKey(-1)
        count+=1
        print(count)