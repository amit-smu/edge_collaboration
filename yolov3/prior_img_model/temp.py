import cv2

file = r"D:\My Drive\yolov3\prior_img_model\data\Images/C7_00001970_prior.png"
image = cv2.imread(file, cv2.IMREAD_UNCHANGED)
print(image.shape)
cv2.imshow("tmp", image)
cv2.waitKey(-1)
