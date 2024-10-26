import os
import cv2
import shutil

if __name__ == "__main__":
    PATH = "../dataset/wildtrack/Images"
    image_list =  os.listdir(PATH)

    count = 0
    for i in image_list:
        if i.__contains__("png"):
            count+=1
            print(f"{i}, {count}")
            img = cv2.imread(f"{PATH}/{i}")
            cv2.imwrite(f"{PATH}/{i[:-4]}.jpg", img)
            os.remove(f"{PATH}/{i}")
