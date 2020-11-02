import cv2

img_dir = r"D:\GitRepo\edge_computing\edge_collaboration\src\spatial_correlation\wildtrack_files\intermediate_frames"
actual_overlap = [6, 4, 908, 1074]
cam_pair = (1, 4)

image = cv2.imread("{}\{}".format(img_dir, "marked_area_cam_r1_c4_f_1995.jpg"))
assert image is not None

# black border
cv2.rectangle(image, (0, 0), (1920, 1080), (0, 0, 0), 10)

# highlight actual overlap area
cv2.rectangle(image, (actual_overlap[0], actual_overlap[1]), (actual_overlap[2], actual_overlap[3]), (0, 255, 0), 10)
cv2.putText(image, "Actual Overlap", (200, actual_overlap[1] + 70), cv2.FONT_HERSHEY_PLAIN, 4.5, (0, 255, 0), 7)

cv2.imshow("image", image)
# cv2.waitKey(-1)
cv2.imwrite("WT_spatial_overlap_weighted.jpg", image)
