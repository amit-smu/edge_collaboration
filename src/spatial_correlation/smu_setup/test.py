import cv2

img_1 = cv2.imread("CAMPUS_spatial_overlap.jpg")
img_2 = cv2.imread("CAMPUS_spatial_overlap_weighted.jpg")

gt_box_coords_vw_1 = [360, 0, 1010, 1080]
scale = 1056 / 1080
gt_box_coords_vw_1 = [int(t * scale) for t in gt_box_coords_vw_1]

img_1 = cv2.rectangle(img_1, (0, 0), (1056, 1056), (0, 0, 255), 30)
img_2 = cv2.rectangle(img_2, (gt_box_coords_vw_1[0], gt_box_coords_vw_1[1]),
                      (gt_box_coords_vw_1[2], gt_box_coords_vw_1[2]), (0, 255, 0), 7)
cv2.imshow("img_1", img_1)
cv2.imshow("img_2", img_2)

cv2.imwrite("CAMPUS_spatial_overlap.jpg", img_1)
cv2.imwrite("CAMPUS_spatial_overlap_weighted.jpg", img_2)
cv2.waitKey(-1)
