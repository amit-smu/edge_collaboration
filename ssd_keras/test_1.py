import cv2
import numpy as np

img = np.full((512, 512, 3), fill_value=114, dtype=np.uint8)
img[100:300, 200:350] = 255
cv2.imshow("image", img)

print(img.shape)
cv2.waitKey(-1)
