import cv2
import numpy as np


def adjust_gamma(image, gamma):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


def apply_brightness_contrast(input_img, brightness, contrast):
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow) / 255
        gamma_b = shadow

        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()

    if contrast != 0:
        f = 131 * (contrast + 127) / (127 * (131 - contrast))
        alpha_c = f
        gamma_c = 127 * (1 - f)

        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf


if __name__ == "__main__":
    filename = r"../../dataset/SMU_Setup/Spatial_Overlap/Sample_Images/viewer4/frame1_13.jpg"
    GAMMA = 1.2
    image = cv2.imread(filename)
    assert image is not None

    image_gamma = adjust_gamma(image, gamma=GAMMA)

    # image sharpening
    kernel = np.array([[0, -1, 0],
                       [-1, 4.9, -1],
                       [0, -1, 0]])
    image_sharp = cv2.filter2D(image_gamma, ddepth=-1, kernel=kernel)

    # image contrast and brightness correctiion
    image_contrast = apply_brightness_contrast(image_gamma, brightness=50, contrast=50)

    cv2.imshow("Old_Image", image)
    cv2.imshow("Gamma_Image", image_gamma)
    cv2.imshow("Contrast_Image", image_contrast)
    cv2.waitKey(-1)
