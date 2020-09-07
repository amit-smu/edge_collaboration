import cv2
from PIL import Image
import os
import numpy as np

IN_DIR = "temp/eval_dump/"
files = os.listdir(IN_DIR)
for f in files:
    print(f)
    # f = "C5_00000290.png"
    f = "C5_00001640.png"

    image = Image.open("{}/{}".format(IN_DIR, f))
    # image = image.convert("RGB")

    img_array = np.array(image, dtype=np.uint8)
    cv2.imshow("array", img_array)
    cv2.waitKey(-1)
    image.show()
    break
