import cv2

img_path = "D:\GitRepo\edge_computing\edge_collaboration\ssd_keras\\temp\\train_dump\C7_00000660.png"
img = cv2.imread(img_path)
cv2.imshow("img", img)
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
cv2.imshow("img_gray", img_gray)
# cv2.imwrite("temp.png", img_gray)
cv2.waitKey(-1)
