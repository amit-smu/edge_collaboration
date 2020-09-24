import cv2
from PIL import Image
import os
import numpy as np

img_name = "frame_7_0066.jpg"
img_dir = "D:\GitRepo\edge_computing\edge_collaboration\dataset\PETS_org\JPEGImages"
image = cv2.imread("{}/{}".format(img_dir, img_name))
SHARED_AREA_RES = (96, 96)
shared_reg_coords = [20, 80, 630, 420]
# cv2.rectangle(image, (shared_region[0], shared_region[1]), (shared_region[2], shared_region[3]), (0, 255, 0), 2)
xmin_org, ymin_org, xmax_org, ymax_org = shared_reg_coords  # in org cam resolution (720x576 for PETS)
xmin_tr = xmin_org
ymin_tr = ymin_org
xmax_tr = xmax_org
ymax_tr = ymax_org
# xmin_tr = int((xmin_org / 700.0) * 512)
# ymin_tr = int((ymin_org / 700.0) * 512)
# xmax_tr = int((xmax_org / 700.0) * 512)
# ymax_tr = int((ymax_org / 700.0) * 512)
# shared_reg_coords_transformed = [xmin_tr, ymin_tr, xmax_tr, ymax_tr]  # coords in 512x512 image size

# compute new resolution for shared area
reg_width = xmax_tr - xmin_tr  # width of shared region in 512x512 image
reg_height = ymax_tr - ymin_tr
reg_width_tr = int((reg_width / 512.0) * SHARED_AREA_RES[0])  # new width as per 224x224 overall resolution
reg_height_tr = int((reg_height / 512.0) * SHARED_AREA_RES[1])
shared_reg_target_res = (reg_width_tr, reg_height_tr)

shared_reg = image[ymin_tr:ymax_tr, xmin_tr:xmax_tr]
temp = cv2.resize(shared_reg, dsize=shared_reg_target_res, interpolation=cv2.INTER_CUBIC)
shared_reg = cv2.resize(temp, dsize=(reg_width, reg_height), interpolation=cv2.INTER_CUBIC)
image[ymin_tr:ymax_tr, xmin_tr:xmax_tr] = shared_reg

cv2.rectangle(image, (xmin_org, ymin_org), (xmax_org, ymax_org), (0, 255, 0), 1)
cv2.putText(image, "96x96", (xmin_org, ymin_org), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
cv2.imshow("image", image)
cv2.imwrite("temp_img.jpg",image)
# cv2.imshow("sh_reg", sh_reg)
cv2.waitKey(-1)
