"""
module to downsample shared region of images as per the given resolution
"""
import cv2
import os

if __name__ == "__main__":
    cam_pair = "1_4"

    in_data_dir = "./test_data/cam_1_4/1056/images/"
    img_dim = (1056.0, 1056.0)

    resolutions = [512, 416, 320, 224, 128]
    spatial_overlap = {
        "1_4": [6, 4, 90, 104],
        "5_7": [91, 142, 693, 510]
    }

    x1, y1, x2, y2 = spatial_overlap[cam_pair]
    print("Cam pair: {}, Spatial Overlap : {}\n".format(cam_pair, spatial_overlap[cam_pair]))
    sh_width = x2 - x1
    sh_height = y2 - y1

    fnames = os.listdir(in_data_dir)
    for res in resolutions:
        print("Resolution: {}\n".format(res))
        out_data_dir = "./test_data/cam_{}/{}/images".format(cam_pair, res)
        for name in fnames:
            print("image name :{}\n".format(name))
            image = cv2.imread("{}/{}".format(in_data_dir, name))
            # assert image.shape == (1056, 1056, 3)

            # extract and downsample the shared region
            shared_region = image[y1:y2, x1:x2]
            sh_reg_new_width = int((sh_width / img_dim[0]) * res)
            sh_reg_new_height = int((sh_height / img_dim[1]) * res)

            temp = cv2.resize(shared_region, dsize=(sh_reg_new_width, sh_reg_new_height), interpolation=cv2.INTER_AREA)
            shared_region = cv2.resize(temp, dsize=(sh_width, sh_height), interpolation=cv2.INTER_CUBIC)
            image[y1:y2, x1:x2] = shared_region

            # save the resulting image
            cv2.imwrite("{}/{}".format(out_data_dir, name), image)
            break
