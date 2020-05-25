import numpy as np
import cv2

view_1_area = np.full((400, 400), 255, dtype=np.uint8)

cv2.imshow("view_1_area", view_1_area)

img = np.zeros([400, 400], dtype=np.uint8)
img.fill(255)  # or img[:] = 255
cv2.imshow("img", img)

cv2.waitKey(-1)
