"""
create a list of images for darknet object detections
"""

import os

if __name__=="__main__":
    INPUT_PATH = "../dataset/wildtrack/Images"
    OUTPUT_PATH = "../../dataset/wildtrack/Images" # relative to darknet directory
    image_names = os.listdir(INPUT_PATH)
    with open("results_and_images/images_wt.txt", 'w') as output:
        for name in image_names:
            if not name.__contains__("jpg"):
                continue
            img_name = "{}/{}".format(OUTPUT_PATH, name)
            print(img_name)
            output.write(img_name +"\n")
            # break
