import cv2

img_dir = "./"
actual_overlap = [28, 101, 617, 492]
cam_pair = (7, 8)

image = cv2.imread("{}\{}".format(img_dir, "view_1_area.jpg"))
assert image is not None

# black border
cv2.rectangle(image, (0, 0), (720, 576), (0, 0, 0), 5)

# highlight actual overlap area
cv2.rectangle(image, (actual_overlap[0], actual_overlap[1]), (actual_overlap[2], actual_overlap[3]), (0, 255, 0), 5)
cv2.putText(image, "Actual Overlap", (150, actual_overlap[3] - 40), cv2.FONT_HERSHEY_PLAIN, 2.5, (0, 255, 0), 2)

cv2.imshow("image", image)
# cv2.waitKey(-1)
cv2.imwrite("spatial_overlap_cam_7_wt_approach.jpg", image)
