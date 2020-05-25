"""
module to resize images and add padding if necessary
"""

import cv2


def resize(dsize, img, padding_value):
    img_org = img.copy()
    img_h, img_w, img_d = img.shape

    desired_h, desired_w = dsize

    # adjust height
    if img_h > desired_h:
        img = cv2.resize(img, dsize=(img_w, desired_h))
    elif img_h < desired_h:
        print("padding")

        img = cv2.copyMakeBorder(src=img, top=0, bottom=desired_h - img_h, left=0, right=0,
                                 borderType=cv2.BORDER_CONSTANT,
                                 value=padding_value)

    # adjust width
    if img_w > desired_w:
        img = cv2.resize(img, dsize=(desired_w, img.shape[0]))
    elif img_w < desired_w:
        print("padding")
        img = cv2.copyMakeBorder(src=img, top=0, bottom=0, left=0, right=desired_w - img_w,
                                 borderType=cv2.BORDER_CONSTANT,
                                 value=padding_value)

    cv2.imshow("org_img", img_org)
    cv2.imshow("modified_img", img)
    cv2.waitKey(-1)

    return img


if __name__ == "__main__":
    img_path = "D:/GitRepo/edge_computing/edge_collaboration/dataset/PETS/JPEGImages/frame_57_0055.jpg"
    image = cv2.imread(img_path)
    assert image is not None

    resize(dsize=(400, 900), img=image, padding_value=(104, 123, 117))
